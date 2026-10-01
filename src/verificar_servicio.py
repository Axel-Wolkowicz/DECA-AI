"""Verifica un despliegue del servicio de inferencia contra el modelo congelado.

    python src/verificar_servicio.py --url http://127.0.0.1:8000
    python src/verificar_servicio.py --url https://<url publica> --token <secreto>

Que el servicio responda no alcanza: hay que saber que responde **lo mismo que el modelo
que se midio**. Este script le manda ECG de validacion cuyos scores se conocen y compara.
Ademas comprueba que el checkpoint sea el congelado, que pida token y que rechace lo que
no es un ECG. Sale con codigo 1 si algo no coincide.

**Solo usa la biblioteca estandar**, a proposito: tiene que correr en el servidor de la
institucion con cualquier python3, sin instalar nada ni tener el venv del proyecto.

Los casos viven en `models/patrones-lr8/verificacion/`, al lado del checkpoint, porque los
scores esperados valen para ese checkpoint y no para otro. Son registros de **validacion**,
nunca de test. Se regeneran con `--generar` (eso si necesita el venv completo y el SSD).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_CASOS = RAIZ / "models" / "patrones-lr8" / "verificacion"
ESPERADO = DIR_CASOS / "esperado.json"

# En otra CPU el mismo float32 puede diferir en el ultimo bit de algunas operaciones. La
# diferencia medida entre caminos de codigo distintos fue 1,65e-4 como maximo (FASES.md,
# sesion del 2026-09-22); en la misma maquina es 0,0. La banda no tiene tolerancia.
TOL_SCORE = 1e-4
TOL_PERCENTIL = 0.05


# --------------------------------------------------------------------------------------
def _pedir(url, metodo="GET", token=None, cuerpo=None, tipo=None, timeout=60):
    """(status, json o None, segundos). No levanta por codigos HTTP de error."""
    pedido = urllib.request.Request(url, data=cuerpo, method=metodo)
    if token:
        pedido.add_header("X-DECA-Token", token)
    if tipo:
        pedido.add_header("Content-Type", tipo)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(pedido, timeout=timeout) as r:
            status, datos = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, datos = e.code, e.read()
    dt = time.perf_counter() - t0
    try:
        return status, json.loads(datos), dt
    except ValueError:
        return status, None, dt


def _multipart(nombre_archivo, datos):
    """El mismo multipart que manda DECA-Back: un campo `archivo`."""
    limite = uuid.uuid4().hex
    cuerpo = (
        f"--{limite}\r\n"
        f'Content-Disposition: form-data; name="archivo"; filename="{nombre_archivo}"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode() + datos + f"\r\n--{limite}--\r\n".encode()
    return cuerpo, f"multipart/form-data; boundary={limite}"


def _ruido():
    """Ruido blanco gaussiano de 12 derivaciones, 10 s a 400 Hz. Semilla fija."""
    rng = random.Random(0)
    señal = [[round(rng.gauss(0, 1), 4) for _ in range(12)] for _ in range(4000)]
    return json.dumps({"frecuencia": 400, "derivaciones": _DERIVACIONES,
                       "senal": señal}).encode()


_DERIVACIONES = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]


# --------------------------------------------------------------------------------------
def verificar(url: str, token: str) -> bool:
    esperado = json.loads(ESPERADO.read_text(encoding="utf-8"))
    url = url.rstrip("/")
    fallas = []

    def chequeo(ok, texto):
        print(f"  {'OK   ' if ok else 'FALLA'} {texto}")
        if not ok:
            fallas.append(texto)

    print(f"Servicio: {url}\n")

    print("Salud y modelo")
    try:
        st, cuerpo, _ = _pedir(f"{url}/salud", timeout=15)
    except (urllib.error.URLError, OSError) as e:
        print(f"  FALLA no se pudo conectar: {e}")
        return False
    chequeo(st == 200 and cuerpo and cuerpo.get("ok"), f"/salud responde 200 (dio {st})")
    sha = (cuerpo or {}).get("sha256", "")
    chequeo(sha == esperado["sha256"],
            f"checkpoint congelado: sha256 {sha or '?'} (esperado {esperado['sha256']})")

    print("\nAutenticacion")
    caso0 = next(iter(esperado["casos"]))
    multipart = _multipart(caso0, (DIR_CASOS / caso0).read_bytes())
    st, _, _ = _pedir(f"{url}/analizar", "POST", None, *multipart)
    chequeo(st == 401, f"/analizar sin token da 401 (dio {st})"
            + ("  <- EL SERVICIO ESTA PUBLICADO SIN AUTENTICACION" if st == 200 else ""))
    st, _, _ = _pedir(f"{url}/contrato", token=token)
    chequeo(st == 200, f"/contrato con el token da 200 (dio {st})"
            + ("  <- el token no coincide con el del servidor" if st == 401 else ""))
    if st == 401:
        print("\nSin un token valido no se pueden probar los analisis.")
        return False

    print("\nAnalisis (ECG de validacion con resultado conocido)")
    tiempos = []
    casos = [(n, (DIR_CASOS / n).read_bytes(), e) for n, e in esperado["casos"].items()]
    casos.append(("ruido blanco", _ruido(), {"error": "no_parece_ecg"}))
    for nombre, datos, esp in casos:
        st, cuerpo, dt = _pedir(f"{url}/analizar", "POST", token, *_multipart(nombre, datos))
        tiempos.append(dt)
        if "error" in esp:
            codigo = ((cuerpo or {}).get("error") or {}).get("codigo")
            chequeo(st == 422 and codigo == esp["error"],
                    f"{nombre}: rechazado con {esp['error']} (dio {st} {codigo})")
            continue
        if st != 200 or not (cuerpo or {}).get("ok"):
            chequeo(False, f"{nombre}: esperaba 200, dio {st} {cuerpo}")
            continue
        a = cuerpo["analisis"]
        d_score = abs(a["score"] - esp["score"])
        d_pct = abs(a["percentil"] - esp["percentil"])
        chequeo(d_score <= TOL_SCORE and a["banda"] == esp["banda"] and d_pct <= TOL_PERCENTIL,
                f"{nombre}: score {a['score']:.6f} (esperado {esp['score']:.6f}, "
                f"dif {d_score:.1e}), banda {a['banda']} (esperada {esp['banda']}), "
                f"percentil {a['percentil']} (esperado {esp['percentil']})")

    tiempos.sort()
    print(f"\nTiempo por pedido (incluye red): mediana {tiempos[len(tiempos) // 2]:.2f} s, "
          f"maximo {tiempos[-1]:.2f} s")
    if fallas:
        print(f"\n{len(fallas)} FALLA(S). El servicio NO reproduce el modelo validado.")
        return False
    print("\nTodo coincide: el servicio reproduce el modelo validado.")
    return True


# --------------------------------------------------------------------------------------
# (fila de fase2_metadata elegida a mano, resultado que se espera de ella)
SELECCION = {
    "code15_2533029.json": "alta",
    "samitrop_247276.json": "alta",
    "code15_2881181.json": "no_alta",
    # Uno de los CODE-15% de exams_part2 con las derivaciones de miembros calculadas con la
    # formula equivocada en la fuente (FASES.md, sesion del 2026-09-28/29).
    "code15_1398291.json": "derivaciones_permutadas",
}


def generar():
    """Regenera los casos y sus resultados esperados con el motor local. Necesita el venv
    completo y fase2_preprocessed.hdf5. Solo hace falta si cambia el checkpoint."""
    import h5py
    import pandas as pd

    sys.path.insert(0, str(RAIZ / "src"))
    from config import FASE2_HDF5, FASE2_METADATA_PATH
    from inferencia import MotorDECA
    from lectura_ecg import DERIVACIONES_CANONICAS, ECGInvalido

    meta = pd.read_parquet(FASE2_METADATA_PATH)
    motor = MotorDECA(device="cpu")
    DIR_CASOS.mkdir(parents=True, exist_ok=True)
    salida = {"sha256": motor.sha256[:16], "casos": {}}

    with h5py.File(FASE2_HDF5, "r") as h5:
        for nombre, que_se_espera in SELECCION.items():
            dataset, record_id = nombre.removesuffix(".json").split("_")
            fila = meta[(meta.dataset == dataset) & (meta.record_id.astype(str) == record_id)]
            assert len(fila) == 1 and fila.iloc[0].split == "val", f"{nombre}: no es 1 fila de val"
            señal = h5["tracings"][int(fila.iloc[0].row_index)]
            # Redondeo a 4 decimales para que los archivos pesen ~1/3. El esperado se calcula
            # sobre el archivo ya redondeado, asi que la comparacion sigue siendo exacta.
            datos = json.dumps({
                "frecuencia": 400,
                "derivaciones": list(DERIVACIONES_CANONICAS),
                "senal": [[round(float(v), 4) for v in fila_s] for fila_s in señal],
            }, separators=(",", ":")).encode()
            (DIR_CASOS / nombre).write_bytes(datos)
            try:
                a = motor.analizar_archivo(datos, "json")
                r = {"score": a["score"], "percentil": a["percentil"], "banda": a["banda"]}
                obtenido = a["banda"]
            except ECGInvalido as e:
                r = {"error": e.codigo}
                obtenido = e.codigo
            assert obtenido == que_se_espera, f"{nombre}: dio {obtenido}, se esperaba {que_se_espera}"
            salida["casos"][nombre] = r
            print(f"{nombre:24s} {len(datos) // 1024:4d} KB  {r}")

    ESPERADO.write_text(json.dumps(salida, indent=2) + "\n", encoding="utf-8")
    print(f"\nescrito {ESPERADO}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--token", default=os.environ.get("DECA_API_TOKEN", ""),
                   help="por defecto, la variable de entorno DECA_API_TOKEN")
    p.add_argument("--generar", action="store_true",
                   help="regenerar los casos (necesita el venv completo y el SSD)")
    args = p.parse_args()

    if args.generar:
        generar()
        sys.exit(0)
    if not args.token:
        sys.exit("Falta el token: --token <secreto> o la variable DECA_API_TOKEN.")
    sys.exit(0 if verificar(args.url, args.token) else 1)
