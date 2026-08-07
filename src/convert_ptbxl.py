"""Convierte PTB-XL de WFDB (.dat/.hea, un archivo por registro) a un HDF5 unico.

PTB-XL es el unico de los tres datasets que no viene en HDF5. Se convierte por dos
motivos (ver ROADMAP, "Formato de archivos"): unificar el formato canonico, y evitar
los ~87.000 archivos chicos que rinden pesimo en el SSD exFAT.

Salida: ptbxl.hdf5 con la misma forma que code15/samitrop, para que el resto del
pipeline lea los tres igual:
    tracings  (N, 5000, 12) float32  - señal en mV
    exam_id   (N,)          int32    - ecg_id de ptbxl_database.csv

Se usa records500 (500 Hz, 10 s) que es la resolucion nativa alta; el resampleo a una
frecuencia comun se hace recien en Fase 2, asi que aca no se toca la señal.

Uso: python src/convert_ptbxl.py
"""
import sys

import h5py
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from config import PTBXL_DATABASE_CSV, PTBXL_DIR, PTBXL_HDF5

FREQ = 500
N_SAMPLES = 5000
# Orden canonico del ROADMAP. PTB-XL ya viene asi, pero se verifica registro por
# registro: si un .hea trajera otro orden, mezclaria derivaciones sin fallar.
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    if not PTBXL_DATABASE_CSV.exists():
        sys.exit(f"No se encontro {PTBXL_DATABASE_CSV}")
    if PTBXL_HDF5.exists():
        sys.exit(
            f"{PTBXL_HDF5} ya existe. Borrarlo a mano si se quiere regenerar "
            "(no se pisa solo para no perder una conversion buena por error)."
        )

    db = pd.read_csv(PTBXL_DATABASE_CSV)
    n = len(db)
    print(f"PTB-XL: {n} registros a convertir desde records500/ ({FREQ} Hz)")

    # Se escribe a un temporal y recien al final se renombra: si el proceso se corta,
    # no queda un .hdf5 a medias que parezca valido.
    tmp_path = PTBXL_HDF5.with_suffix(".hdf5.tmp")
    con_nan = []

    with h5py.File(tmp_path, "w") as f:
        tracings = f.create_dataset(
            "tracings",
            shape=(n, N_SAMPLES, 12),
            dtype="float32",
            chunks=(1, N_SAMPLES, 12),
        )
        f.create_dataset("exam_id", data=db["ecg_id"].to_numpy(dtype="int32"))
        f.attrs["frecuencia"] = FREQ
        f.attrs["duracion"] = N_SAMPLES / FREQ
        f.attrs["leads"] = LEADS
        f.attrs["fuente"] = "records500 (WFDB) de PTB-XL 1.0.3"

        for i, filename in enumerate(tqdm(db["filename_hr"], total=n, unit="reg")):
            # filename_hr viene como 'records500/00000/00001_hr' (sin extension)
            rec = wfdb.rdrecord(str(PTBXL_DIR / filename))

            if rec.sig_name != LEADS:
                sys.exit(f"\n{filename}: orden de derivaciones inesperado {rec.sig_name}")
            if rec.p_signal.shape != (N_SAMPLES, 12):
                sys.exit(f"\n{filename}: forma inesperada {rec.p_signal.shape}")

            if np.isnan(rec.p_signal).any():
                con_nan.append(int(db["ecg_id"].iloc[i]))

            tracings[i] = rec.p_signal.astype("float32")

    tmp_path.replace(PTBXL_HDF5)

    size_gb = PTBXL_HDF5.stat().st_size / 1024**3
    print(f"\nOK {PTBXL_HDF5} ({size_gb:.2f} GB, {n} registros)")
    if con_nan:
        # No se imputan ni descartan aca: es una decision de la fase de limpieza.
        print(f"ADVERTENCIA: {len(con_nan)} registros con NaN en la señal, p.ej. {con_nan[:5]}")


if __name__ == "__main__":
    main()
