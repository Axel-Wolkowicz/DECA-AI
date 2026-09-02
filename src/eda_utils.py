"""Utilidades compartidas por los notebooks de Fase 1 (notebooks/0*_*.ipynb).

Centraliza la resolucion de rutas y la carga de señales para que los 4 notebooks de
EDA no repitan la misma logica de mapeo dataset -> archivo HDF5 que ya vive en
build_metadata.py.
"""
import h5py
import numpy as np
import pandas as pd

from config import CHALLENGE2021_HDF5, CODE15_DIR, METADATA_PATH, PTBXL_HDF5, SAMITROP_HDF5

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def cargar_metadata() -> pd.DataFrame:
    return pd.read_parquet(METADATA_PATH)


def path_para(dataset: str, source_file: str):
    """Resuelve la ruta real del HDF5 a partir de dataset + source_file de metadata."""
    if dataset == "code15":
        return CODE15_DIR / source_file
    if dataset == "samitrop":
        return SAMITROP_HDF5
    if dataset == "ptbxl":
        return PTBXL_HDF5
    if dataset == "challenge2021":
        return CHALLENGE2021_HDF5
    raise ValueError(f"dataset desconocido: {dataset}")


def cargar_senal(row: pd.Series) -> np.ndarray:
    """Devuelve la señal (n_muestras, 12) para una fila de metadata.parquet."""
    path = path_para(row["dataset"], row["source_file"])
    with h5py.File(path, "r") as f:
        return f["tracings"][row["row_index"]]


def cargar_senales_lote(rows: pd.DataFrame) -> list[np.ndarray]:
    """Carga varias señales agrupando por archivo, para no reabrir el mismo HDF5
    una vez por fila (relevante para CODE-15%, que esta partido en 18 archivos)."""
    señales = [None] * len(rows)
    por_archivo: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows.itertuples(index=False)):
        key = (row.dataset, row.source_file)
        por_archivo.setdefault(key, []).append(i)

    for (dataset, source_file), idxs in por_archivo.items():
        path = path_para(dataset, source_file)
        with h5py.File(path, "r") as f:
            tracings = f["tracings"]
            for i in idxs:
                row_index = rows.iloc[i]["row_index"]
                señales[i] = tracings[row_index]
    return señales
