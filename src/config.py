"""Configuracion central de rutas. No hardcodear D:/DECA-DATASETS en otro lado."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DECA_DATA_DIR", "D:/DECA-DATASETS"))

CODE15_DIR = DATA_DIR / "code15"
SAMITROP_DIR = DATA_DIR / "samitrop"
PTBXL_DIR = DATA_DIR / "ptbxl"
METADATA_PATH = DATA_DIR / "metadata.parquet"

CODE15_LABELS_CSV = CODE15_DIR / "code15_chagas_labels.csv"
CODE15_EXAMS_CSV = CODE15_DIR / "exams.csv"
SAMITROP_EXAMS_CSV = SAMITROP_DIR / "exams.csv"
SAMITROP_HDF5 = SAMITROP_DIR / "exams.hdf5"
PTBXL_DATABASE_CSV = PTBXL_DIR / "ptbxl_database.csv"
PTBXL_HDF5 = PTBXL_DIR / "ptbxl.hdf5"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"No se encontro DATA_DIR={DATA_DIR}. "
        "Verificar que el SSD este conectado o setear la variable de entorno DECA_DATA_DIR."
    )
