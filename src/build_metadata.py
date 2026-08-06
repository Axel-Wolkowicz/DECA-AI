"""Consolida labels/demografia de los 3 datasets en un metadata.parquet unico.

Columnas de salida:
    record_id   - id de registro, unico dentro del dataset de origen
    dataset     - 'code15' | 'samitrop' | 'ptbxl'
    source_file - archivo HDF5 (o carpeta WFDB) donde vive la señal
    row_index   - posicion de la fila dentro de source_file (indexa tracings[row_index])
    edad
    sexo        - 'M' | 'F'
    frecuencia  - Hz
    duracion    - segundos
    chagas_label - True/False/None (None = sin dato)
    confianza   - 'weak' (autorreportada) | 'strong' (serologica) | 'negativo-presunto'

Uso: python src/build_metadata.py
"""
import h5py
import numpy as np
import pandas as pd

from config import (
    CODE15_DIR,
    CODE15_EXAMS_CSV,
    CODE15_LABELS_CSV,
    METADATA_PATH,
    PTBXL_DATABASE_CSV,
    SAMITROP_EXAMS_CSV,
    SAMITROP_HDF5,
)

CODE15_FREQ = 400
CODE15_N_PARTS = 18


def build_code15():
    exams = pd.read_csv(CODE15_EXAMS_CSV)
    labels = pd.read_csv(CODE15_LABELS_CSV)[["exam_id", "chagas"]]

    # index exam_id -> row_index dentro de cada exams_part{n}.hdf5 (solo se lee el
    # dataset 'exam_id', no las señales -> rapido)
    row_index = {}
    for n in range(CODE15_N_PARTS):
        path = CODE15_DIR / f"exams_part{n}.hdf5"
        with h5py.File(path, "r") as f:
            for i, eid in enumerate(f["exam_id"][:]):
                row_index[int(eid)] = i

    df = exams.merge(labels, on="exam_id", how="left")
    missing = df["chagas"].isna().sum()
    if missing:
        print(f"CODE-15%: {missing}/{len(df)} registros sin label de chagas (se descartan)")
    df = df.dropna(subset=["chagas"])

    df["row_index"] = df["exam_id"].map(row_index)
    unmatched = df["row_index"].isna().sum()
    if unmatched:
        print(f"CODE-15%: {unmatched} registros en exams.csv sin tracing correspondiente (se descartan)")
    df = df.dropna(subset=["row_index"])
    df["row_index"] = df["row_index"].astype(int)

    return pd.DataFrame({
        "record_id": df["exam_id"].astype(str),
        "dataset": "code15",
        "source_file": df["trace_file"],
        "row_index": df["row_index"],
        "edad": df["age"],
        "sexo": np.where(df["is_male"], "M", "F"),
        "frecuencia": CODE15_FREQ,
        "duracion": 4096 / CODE15_FREQ,
        "chagas_label": df["chagas"].astype(bool),
        "confianza": "weak",
    })


def build_samitrop():
    exams = pd.read_csv(SAMITROP_EXAMS_CSV)
    with h5py.File(SAMITROP_HDF5, "r") as f:
        n_tracings = f["tracings"].shape[0]

    if len(exams) != n_tracings:
        raise ValueError(
            f"SaMi-Trop: exams.csv tiene {len(exams)} filas pero tracings tiene {n_tracings}. "
            "El mapeo row_index=posicion asume orden identico; revisar antes de continuar."
        )

    return pd.DataFrame({
        "record_id": exams["exam_id"].astype(str),
        "dataset": "samitrop",
        "source_file": "exams.hdf5",
        "row_index": np.arange(n_tracings),
        "edad": exams["age"],
        "sexo": np.where(exams["is_male"], "M", "F"),
        "frecuencia": 400,  # ver ROADMAP: SaMi-Trop nativo a 400 Hz
        "duracion": 4096 / 400,
        "chagas_label": True,  # cohorte de pacientes con Chagas (serologia)
        "confianza": "strong",
    })


def build_ptbxl():
    if not PTBXL_DATABASE_CSV.exists():
        print(f"PTB-XL: no encontrado {PTBXL_DATABASE_CSV} todavia (falta descargar/convertir) -> se omite")
        return None

    db = pd.read_csv(PTBXL_DATABASE_CSV)
    return pd.DataFrame({
        "record_id": db["ecg_id"].astype(str),
        "dataset": "ptbxl",
        "source_file": db["filename_hr"],
        "row_index": 0,  # PTB-XL se convierte a un hdf5 por registro o consolidado; TODO al convertir
        "edad": db["age"],
        "sexo": np.where(db["sex"] == 0, "M", "F"),
        "frecuencia": 500,
        "duracion": 10.0,
        "chagas_label": False,  # region no endemica, negativo por presuncion
        "confianza": "negativo-presunto",
    })


def main():
    parts = [build_code15(), build_samitrop()]
    ptbxl = build_ptbxl()
    if ptbxl is not None:
        parts.append(ptbxl)

    metadata = pd.concat(parts, ignore_index=True)
    metadata.to_parquet(METADATA_PATH, index=False)

    print(f"\nmetadata.parquet -> {METADATA_PATH}")
    print(metadata.groupby(["dataset", "confianza"]).agg(
        n=("record_id", "count"), positivos=("chagas_label", "sum")
    ))
    if ptbxl is None:
        print("\nADVERTENCIA: PTB-XL no incluido en esta corrida. Volver a correr cuando este disponible.")


if __name__ == "__main__":
    main()
