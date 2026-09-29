"""Mide los controles de `plausibilidad.py` por el mismo camino que usa el servicio.

Dos preguntas, y la primera es la que importa:

1. **Falso rechazo**: de los ECG reales de validacion (las 4 fuentes, señal cruda, todos
   los que pasaron la Fase 2), cuantos rechazaria el servicio y por que. Cada rechazo de un
   ECG real es un paciente al que no se le da resultado. Se reporta ademas la etiqueta de
   Chagas de los rechazados: si el control se comiera desproporcionadamente positivos,
   estaria recortando justo la poblacion que importa.
2. **Deteccion**: sobre basura sintetica y sobre ECG reales con derivaciones permutadas,
   cuanto ataja.

Solo validacion, nunca test (CLAUDE.md, Fase 3): los umbrales se fijaron con estos datos.
Usa `MotorDECA._preparar`, o sea el codigo del servicio y no una copia, sobre una
`LecturaECG` armada con la señal cruda del corpus en orden canonico.

Uso:
    python src/validar_plausibilidad.py [--limit N] [--salida medidas.parquet]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import config
from eda_utils import cargar_senales_lote
from inferencia import MotorDECA
from lectura_ecg import DERIVACIONES_CANONICAS, ECGInvalido, LecturaECG


def medir(motor: MotorDECA, señal: np.ndarray, fs: float, derivadas=()) -> dict:
    lectura = LecturaECG(señal=señal, frecuencia=fs, formato="validacion",
                         derivaciones_origen=list(DERIVACIONES_CANONICAS),
                         derivadas=list(derivadas))
    try:
        _, calidad = motor._preparar(lectura)
        coh = calidad["coherencia_miembros"] or {}
        return {"resultado": "ok", "concentracion_qrs": calidad["concentracion_qrs"],
                "r2": coh.get("r2"), "coseno": coh.get("coseno")}
    except ECGInvalido as e:
        return {"resultado": e.codigo, "detalle": e.detalle}


def derivar_miembros(x: np.ndarray) -> np.ndarray:
    """Pisa III/aVR/aVL/aVF con las identidades: simula un archivo de 8 derivaciones."""
    x = x.copy()
    i, ii = x[:, 0], x[:, 1]
    x[:, 2], x[:, 3], x[:, 4], x[:, 5] = ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2
    return x


def basura(rng: np.random.Generator, reales: list[np.ndarray], n: int):
    """(tipo, señal, fs, derivadas) para cada caso sintetico."""
    ocho = ["III", "aVR", "aVL", "aVF"]
    t = np.arange(4000) / 400
    for k in range(n):
        blanco = rng.standard_normal((4000, 12))
        rosa = np.fft.irfft(np.fft.rfft(rng.standard_normal((4000, 12)), axis=0)
                            / np.sqrt(np.arange(1, 2002))[:, None], n=4000, axis=0)
        paseo = np.cumsum(rng.standard_normal((4000, 12)), axis=0)
        yield "ruido_blanco", blanco, 400, []
        yield "ruido_blanco_8deriv", derivar_miembros(blanco), 400, ocho
        yield "ruido_rosa", rosa, 400, []
        yield "ruido_rosa_8deriv", derivar_miembros(rosa), 400, ocho
        yield "random_walk", paseo, 400, []
        yield "random_walk_8deriv", derivar_miembros(paseo), 400, ocho
        senos = (np.sin(2 * np.pi * rng.uniform(0.5, 5) * t)[:, None] * rng.uniform(0.5, 2, 12)
                 + 0.01 * rng.standard_normal((4000, 12)))
        yield "senos", senos, 400, []

        x = reales[k % len(reales)]
        s = x.copy(); s[:, [3, 4]] = s[:, [4, 3]]
        yield "ecg_swap_aVR_aVL", s, 400, []
        s = x.copy(); s[:, [0, 1]] = s[:, [1, 0]]
        yield "ecg_swap_I_II", s, 400, []
        perm = rng.permutation(6)
        while (perm == np.arange(6)).all():
            perm = rng.permutation(6)
        s = x.copy(); s[:, :6] = s[:, perm]
        yield "ecg_perm_miembros", s, 400, []
        s = x.copy(); s[:, 6:] = s[:, 6 + rng.permutation(6)]
        yield "ecg_perm_precordiales (indetectable)", s, 400, []


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--limit", type=int, default=None, help="registros de val por dataset")
    p.add_argument("--n-sinteticos", type=int, default=300)
    p.add_argument("--salida", default=None, help="parquet con la medida de cada registro")
    args = p.parse_args()

    motor = MotorDECA(device="cpu")
    f2 = pd.read_parquet(config.FASE2_METADATA_PATH)
    meta = pd.read_parquet(config.METADATA_PATH)[
        ["dataset", "record_id", "frecuencia", "source_file", "row_index"]]
    # source_file/row_index de fase2_metadata apuntan al HDF5 preprocesado; la señal cruda
    # se ubica con los de metadata.parquet. Merge por (dataset, record_id), nunca solo id.
    val = (f2[f2.split == "val"].drop(columns=["source_file", "row_index"])
           .merge(meta, on=["dataset", "record_id"], validate="one_to_one"))
    if args.limit:
        val = pd.concat([g.sample(min(len(g), args.limit), random_state=0)
                         for _, g in val.groupby("dataset")])

    filas, t0 = [], time.time()
    for (dataset, _), g in val.groupby(["dataset", "source_file"]):
        for row, s in zip(g.itertuples(index=False), cargar_senales_lote(g)):
            r = medir(motor, np.asarray(s, dtype=np.float32), row.frecuencia)
            r.update(dataset=dataset, record_id=row.record_id, chagas=row.chagas_label)
            filas.append(r)
        print(f"  {dataset:<14} {len(filas):>6,} registros  {time.time() - t0:5.0f}s", flush=True)
    reales = pd.DataFrame(filas)

    print("\n=== 1. Falso rechazo sobre ECG reales de validacion ===")
    tabla = reales.assign(rechazado=reales.resultado != "ok").groupby("dataset").agg(
        n=("resultado", "size"), rechazados=("rechazado", "sum"))
    tabla["tasa"] = (tabla.rechazados / tabla.n * 100).round(3).astype(str) + "%"
    print(tabla.to_string())
    print("\npor motivo:")
    print(reales[reales.resultado != "ok"].groupby(["dataset", "resultado"]).size().to_string())
    rech = reales[reales.resultado != "ok"]
    con = reales[reales.chagas.notna()]
    print(f"\nChagas+ entre rechazados con etiqueta: {rech.chagas.dropna().astype(float).mean():.3%} "
          f"(n={rech.chagas.notna().sum()}) vs {con.chagas.astype(float).mean():.3%} en val")
    ok = reales[reales.resultado == "ok"]
    print("\nconcentracion_qrs de los aceptados, cuantiles:")
    print(ok.groupby("dataset").concentracion_qrs.quantile([0, .0005, .001, .01, .5])
          .unstack().round(3).to_string())

    print("\n=== 2. Deteccion sobre casos sinteticos ===")
    rng = np.random.default_rng(0)
    code15 = val[val.dataset == "code15"]
    base = code15.sample(min(len(code15), 200), random_state=1)
    ecgs = []
    for row, s in zip(base.itertuples(index=False), cargar_senales_lote(base)):
        s = np.asarray(s, dtype=np.float32)
        if medir(motor, s, 400)["resultado"] == "ok":   # partir de ECG que el servicio acepta
            ecgs.append(s)
    sint = pd.DataFrame([{"tipo": tipo, **medir(motor, x.astype(np.float32), fs, der)}
                         for tipo, x, fs, der in basura(rng, ecgs, args.n_sinteticos)])
    det = sint.assign(rechazado=sint.resultado != "ok").groupby("tipo", sort=False).agg(
        n=("resultado", "size"), rechazado=("rechazado", "mean"))
    det["rechazado"] = (det.rechazado * 100).round(1).astype(str) + "%"
    det["motivos"] = sint.groupby("tipo", sort=False).resultado.agg(
        lambda s: dict(s.value_counts()))
    print(det.to_string())

    if args.salida:
        reales.to_parquet(args.salida)
        print(f"\nmedidas por registro -> {args.salida}")


if __name__ == "__main__":
    main()
