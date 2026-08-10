"""Configuracion central de rutas. La ubicacion del SSD no se hardcodea en ningun lado.

El SSD externo cambia de letra segun el puerto USB donde se enchufe (ya monto en D: y
en E:), asi que fijar una letra en el codigo garantiza que en algun momento falle. La
ruta se resuelve en este orden:

  1. Variable de entorno DECA_DATA_DIR (gana siempre; util para apuntar a una copia).
  2. Autodeteccion: se recorren las unidades montadas buscando una carpeta de datos
     valida (ver CARPETAS_CANDIDATAS / _es_valida).
  3. Si no se encuentra nada, error con instrucciones.

Para ver que ruta se esta resolviendo: python src/config.py
"""
import os
from pathlib import Path

# Una carpeta cuenta como carpeta de datos solo si tiene los tres datasets adentro.
# Chequear la existencia del directorio a secas no alcanza: una carpeta vacia con el
# nombre correcto haria que todo el pipeline fallara mucho mas tarde y peor.
SUBDIRS_REQUERIDOS = ("code15", "samitrop", "ptbxl")
CARPETAS_CANDIDATAS = ("DECA-datasets", "DECA-DATASETS")


def _es_valida(path: Path) -> bool:
    return path.is_dir() and all((path / sub).is_dir() for sub in SUBDIRS_REQUERIDOS)


def _unidades():
    """Raices donde buscar. En Windows, las unidades montadas; en POSIX, los puntos
    de montaje habituales de medios extraibles."""
    if hasattr(os, "listdrives"):  # Windows, Python >= 3.12
        try:
            return [Path(d) for d in os.listdrives()]
        except OSError:
            pass
    return [Path("/media"), Path("/mnt"), Path("/Volumes"), Path.home()]


def _autodetectar() -> Path | None:
    for unidad in _unidades():
        for nombre in CARPETAS_CANDIDATAS:
            candidata = unidad / nombre
            try:
                if _es_valida(candidata):
                    return candidata
            except OSError:
                continue  # unidad sin medio insertado, sin permisos, etc.
    return None


def resolver_data_dir() -> Path:
    env = os.environ.get("DECA_DATA_DIR")
    if env:
        path = Path(env)
        if not _es_valida(path):
            faltantes = [s for s in SUBDIRS_REQUERIDOS if not (path / s).is_dir()]
            raise FileNotFoundError(
                f"DECA_DATA_DIR={path} no es una carpeta de datos valida "
                f"(faltan subcarpetas: {', '.join(faltantes) or 'la carpeta no existe'}). "
                "Corregir la variable o desetearla para que se autodetecte el SSD."
            )
        return path

    detectada = _autodetectar()
    if detectada is None:
        raise FileNotFoundError(
            "No se encontro la carpeta de datos en ninguna unidad montada.\n"
            f"Se busco {' o '.join(CARPETAS_CANDIDATAS)} con las subcarpetas "
            f"{', '.join(SUBDIRS_REQUERIDOS)} adentro.\n"
            "Verificar que el SSD externo este conectado, o setear la ruta a mano:\n"
            '  PowerShell:  $env:DECA_DATA_DIR = "D:/DECA-datasets"\n'
            "  bash:        export DECA_DATA_DIR=/ruta/al/ssd/DECA-datasets"
        )
    return detectada


DATA_DIR = resolver_data_dir()

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

# Salida de Fase 2 (src/preprocess.py): señal unificada (N, 2800, 12) a 400 Hz,
# ya recortada y normalizada, lista para copiar a la maquina de entrenamiento.
FASE2_HDF5 = DATA_DIR / "fase2_preprocessed.hdf5"
FASE2_METADATA_PATH = DATA_DIR / "fase2_metadata.parquet"


if __name__ == "__main__":
    origen = "DECA_DATA_DIR" if os.environ.get("DECA_DATA_DIR") else "autodeteccion"
    print(f"DATA_DIR = {DATA_DIR}  (via {origen})")
    for nombre, path in [
        ("code15", CODE15_DIR),
        ("samitrop", SAMITROP_DIR),
        ("ptbxl", PTBXL_DIR),
        ("metadata.parquet", METADATA_PATH),
    ]:
        print(f"  {'OK ' if path.exists() else 'FALTA'} {nombre:<16} {path}")
