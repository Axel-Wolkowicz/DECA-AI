"""Fase 2: unifica los 3 datasets a una señal comun (N, 2800, 12) a 400 Hz.

Por registro:
    1. Se descuenta el padding de ceros (tramos contiguos al inicio/final donde las 12
       derivaciones son exactamente 0 -- ver notebooks/04_calidad_senal.ipynb).
    2. Si lo que queda de señal real dura menos de 7.0s, se DESCARTA el registro entero
       (no se rellena con ceros: eso reintroduciria el atajo de padding que este diseño
       busca eliminar -- ver FASES.md, Fase 4, "saber de que dataset viene un registro
       equivale a saber la etiqueta"). Medido el 2026-08-10: pasa en 5.27% de SaMi-Trop
       (86/1.631, incluye positivos fuertes que se pierden) y ~1.1% de CODE-15%
       (incluye al menos un registro totalmente vacio). PTB-XL: 0%.
    3. Si la señal real contiene valores no finitos (NaN o inf) se DESCARTA el registro.
       Hay corrupcion en los archivos de origen: medido el 2026-08-10 sobre el corpus
       completo, el registro 2858700 de code15/exams_part2.hdf5 trae 23 celdas NaN y
       1.544 muestras de magnitud ~1e38 mezcladas con señal normal de ~0.03 mV. Es 1 de
       362.364 (0.0003%), pero un solo NaN que llegue al entrenamiento vuelve NaN la
       perdida, los gradientes y despues todos los pesos de la red.
    4. PTB-XL (500 Hz) se resamplea a 400 Hz con resample_poly (factor exacto 4/5). Los
       otros dos ya estan a 400 Hz nativo.
    5. Se recorta una ventana de 7.0s = 2.800 muestras, CENTRADA sobre la señal real (no
       desde el inicio): evita las puntas, donde suele haber transiente de contacto del
       electrodo asentandose (visible en notebooks/01_eda_senales.ipynb).
    6. Z-score por registro y por derivacion (no global): elimina diferencia de escala
       entre datasets. Si una derivacion es plana (std ~0, electrodo desconectado -- raro,
       ver notebooks/04_calidad_senal.ipynb), se deja sin escalar en vez de dividir por 0.

Ademas asigna train/val/test por paciente (ver split_patients.py) antes de procesar, para
poder loggear cuantos registros se pierden por split.

Uso: python src/preprocess.py [--limit N]
"""
import argparse
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from config import FASE2_HDF5, FASE2_METADATA_PATH
from eda_utils import cargar_metadata, path_para
from split_patients import asignar_split

OUT_FREQ = 400
WINDOW_SEC = 7.0
WINDOW_SAMPLES = int(WINDOW_SEC * OUT_FREQ)  # 2800


def recortar_padding(señal: np.ndarray) -> np.ndarray:
    ceros = np.all(señal == 0, axis=1)
    n = len(ceros)
    inicio = 0
    while inicio < n and ceros[inicio]:
        inicio += 1
    fin = 0
    while fin < n - inicio and ceros[n - 1 - fin]:
        fin += 1
    return señal[inicio : n - fin] if fin > 0 else señal[inicio:]


def procesar_registro(señal: np.ndarray, frecuencia_nativa: int) -> tuple[np.ndarray | None, str]:
    """Devuelve (ventana normalizada, "ok") o (None, motivo del descarte)."""
    tramo = recortar_padding(señal)
    if tramo.shape[0] / frecuencia_nativa < WINDOW_SEC:
        return None, "corto"

    # Se chequea sobre el tramo entero y antes de resamplear: resample_poly es una
    # convolucion, asi que un solo NaN fuera de la ventana se esparciria hacia adentro.
    if not np.isfinite(tramo).all():
        return None, "corrupto"

    if frecuencia_nativa != OUT_FREQ:
        g = np.gcd(int(frecuencia_nativa), OUT_FREQ)
        tramo = resample_poly(tramo, up=OUT_FREQ // g, down=int(frecuencia_nativa) // g, axis=0)

    n = tramo.shape[0]
    if n < WINDOW_SAMPLES:
        return None, "corto"  # margen ante redondeo del resampleo, no deberia pasar

    inicio = (n - WINDOW_SAMPLES) // 2
    ventana = tramo[inicio : inicio + WINDOW_SAMPLES].astype(np.float32)

    media = ventana.mean(axis=0)
    desvio = ventana.std(axis=0)
    desvio[desvio < 1e-8] = 1.0  # derivacion plana: no dividir por 0, dejarla sin escalar
    return (ventana - media) / desvio, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo los primeros N registros (para probar)")
    args = parser.parse_args()

    meta = cargar_metadata()
    meta["split"] = asignar_split(meta)
    if args.limit:
        meta = meta.iloc[: args.limit].copy()

    n_total = len(meta)
    print(f"Procesando {n_total} registros -> {FASE2_HDF5}")

    filas_out = []
    descartados = Counter()  # (dataset, motivo) -> cantidad
    contador = 0

    with h5py.File(FASE2_HDF5, "w") as fout:
        tracings_out = fout.create_dataset(
            "tracings",
            shape=(n_total, WINDOW_SAMPLES, 12),
            maxshape=(n_total, WINDOW_SAMPLES, 12),
            dtype="float32",
            chunks=(1, WINDOW_SAMPLES, 12),
        )
        fout.attrs["frecuencia"] = OUT_FREQ
        fout.attrs["duracion"] = WINDOW_SEC
        fout.attrs["normalizacion"] = "z-score por registro y por derivacion"

        for (dataset, source_file), grupo in tqdm(meta.groupby(["dataset", "source_file"]), desc="archivos"):
            path = path_para(dataset, source_file)
            with h5py.File(path, "r") as fin:
                tracings_in = fin["tracings"]
                for idx, row in grupo.iterrows():
                    señal = tracings_in[row["row_index"]]
                    procesada, motivo = procesar_registro(señal, row["frecuencia"])
                    if procesada is None:
                        descartados[(dataset, motivo)] += 1
                        continue

                    tracings_out[contador] = procesada
                    filas_out.append({
                        "record_id": row["record_id"],
                        "dataset": row["dataset"],
                        "patient_id": row["patient_id"],
                        "source_file": FASE2_HDF5.name,
                        "row_index": contador,
                        "edad": row["edad"],
                        "sexo": row["sexo"],
                        "chagas_label": row["chagas_label"],
                        "confianza": row["confianza"],
                        "split": row["split"],
                    })
                    contador += 1

        tracings_out.resize((contador, WINDOW_SAMPLES, 12))

    metadata_out = pd.DataFrame(filas_out)
    metadata_out.to_parquet(FASE2_METADATA_PATH, index=False)

    print(f"\n{contador}/{n_total} registros procesados y guardados en {FASE2_HDF5}")
    print(f"\nDescartados ({sum(descartados.values())} en total):")
    for (ds, motivo), cantidad in sorted(descartados.items()):
        detalle = f"<{WINDOW_SEC}s de señal real" if motivo == "corto" else "valores no finitos en la fuente"
        print(f"  {ds:<10} {motivo:<9} {cantidad:>6}  ({detalle})")
    print("\nPor split y dataset:")
    print(metadata_out.groupby(["split", "dataset"]).size().unstack(fill_value=0))
    print("\n% positivos por split:")
    print(metadata_out.groupby("split")["chagas_label"].mean().mul(100).round(2))


if __name__ == "__main__":
    main()
