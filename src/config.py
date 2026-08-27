"""Configuracion central de rutas. La ubicacion del SSD no se hardcodea en ningun lado.

Funciona igual en Windows y en Linux, a proposito: las Fases 0-2 (armado del dataset)
corren en la notebook con Windows, pero la Fase 4 (entrenamiento) corre en la maquina con
la RTX 4090, que tiene Ubuntu, con el mismo SSD enchufado. Nada de esto asume separadores
de ruta, letras de unidad ni puntos de montaje concretos.

La ruta se resuelve en este orden:

  1. Variable de entorno DECA_DATA_DIR (gana siempre; es lo mas rapido y explicito, y en
     la maquina de entrenamiento conviene fijarla en el ~/.bashrc y olvidarse).
  2. Autodeteccion: se recorren los puntos de montaje habituales de cada sistema buscando
     una carpeta de datos valida (ver CARPETAS_CANDIDATAS / _es_valida).
  3. Si no se encuentra nada, error con instrucciones para los dos sistemas.

Para ver que ruta se esta resolviendo: python src/config.py
"""
import os
from pathlib import Path

# Una carpeta cuenta como carpeta de datos solo si tiene los tres datasets adentro.
# Chequear la existencia del directorio a secas no alcanza: una carpeta vacia con el
# nombre correcto haria que todo el pipeline fallara mucho mas tarde y peor.
SUBDIRS_REQUERIDOS = ("code15", "samitrop", "ptbxl")
CARPETAS_CANDIDATAS = ("DECA-datasets", "DECA-DATASETS", "deca-datasets")

# Profundidad de busqueda bajo cada raiz. En Linux el automontaje de un medio externo
# queda en /media/<usuario>/<etiqueta>/, asi que la carpeta puede estar dos niveles abajo
# de /media. En Windows las unidades son raices y con 0 alcanzaria.
PROFUNDIDAD_BUSQUEDA = 2


def _es_valida(path: Path) -> bool:
    return path.is_dir() and all((path / sub).is_dir() for sub in SUBDIRS_REQUERIDOS)


def _raices():
    """Puntos de partida de la busqueda, segun el sistema."""
    if hasattr(os, "listdrives"):  # Windows, Python >= 3.12
        try:
            return [Path(d) for d in os.listdrives()]
        except OSError:
            pass
    # POSIX: puntos de montaje de medios extraibles (/media/<user>/<etiqueta> en Ubuntu,
    # /run/media/<user>/<etiqueta> en Fedora/Arch, /Volumes en macOS), mas los lugares
    # razonables para una copia local en disco interno.
    return [
        Path("/media"),
        Path("/run/media"),
        Path("/mnt"),
        Path("/Volumes"),
        Path("/data"),
        Path.home(),
    ]


def _candidatas(raiz: Path, profundidad: int):
    """La raiz y sus subdirectorios hasta `profundidad` niveles. Se ignoran los ocultos
    (en $HOME son cientos y ninguno va a tener el SSD adentro)."""
    yield raiz
    if profundidad <= 0:
        return
    try:
        subdirs = [d for d in raiz.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return  # unidad sin medio insertado, sin permisos, ruta inexistente
    for sub in subdirs:
        yield from _candidatas(sub, profundidad - 1)


def _autodetectar() -> Path | None:
    for raiz in _raices():
        for candidata in _candidatas(raiz, PROFUNDIDAD_BUSQUEDA):
            try:
                # El punto de montaje puede SER la carpeta de datos (/mnt/deca con code15/
                # adentro) o contenerla (/media/user/SSD/DECA-datasets). Se prueban los dos.
                if _es_valida(candidata):
                    return candidata
                for nombre in CARPETAS_CANDIDATAS:
                    if _es_valida(candidata / nombre):
                        return candidata / nombre
            except OSError:
                continue
    return None


def resolver_data_dir() -> Path:
    env = os.environ.get("DECA_DATA_DIR")
    if env:
        path = Path(env).expanduser()
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
            "No se encontro la carpeta de datos en ningun punto de montaje.\n"
            f"Se busco {' o '.join(CARPETAS_CANDIDATAS)} (hasta {PROFUNDIDAD_BUSQUEDA} "
            f"niveles bajo cada raiz) con las subcarpetas {', '.join(SUBDIRS_REQUERIDOS)} "
            "adentro.\n"
            "Verificar que el SSD externo este conectado, o setear la ruta a mano:\n"
            '  PowerShell:  $env:DECA_DATA_DIR = "D:/DECA-datasets"\n'
            "  bash:        export DECA_DATA_DIR=/media/$USER/SSD/DECA-datasets"
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

# Challenge 2021 (ver ROADMAP, "Datasets candidatos a incorporar"): sin label de Chagas,
# aporta labels de patron ECG (SNOMED-CT) para las cabezas auxiliares. No entra en
# METADATA_PATH/build_metadata.py todavia -- integrarlo a Fase 4 (cabeza de Chagas
# enmascarada) es una decision de arquitectura pendiente, ver FASES.md.
CHALLENGE2021_DIR = DATA_DIR / "challenge2021"
CHALLENGE2021_RAW_DIR = CHALLENGE2021_DIR / "training"
CHALLENGE2021_HDF5 = CHALLENGE2021_DIR / "challenge2021.hdf5"
CHALLENGE2021_LABELS_CSV = CHALLENGE2021_DIR / "challenge2021_labels.csv"

# Salida de Fase 2 (src/preprocess.py): señal unificada (N, 2800, 12) a 400 Hz,
# ya recortada y normalizada, lista para copiar a la maquina de entrenamiento.
FASE2_HDF5 = DATA_DIR / "fase2_preprocessed.hdf5"
FASE2_METADATA_PATH = DATA_DIR / "fase2_metadata.parquet"

# Fase 4: checkpoints y metricas de cada corrida. Van al SSD junto con los datos, no al
# repo (pesan decenas de MB por corrida).
MODELOS_DIR = DATA_DIR / "modelos"


if __name__ == "__main__":
    import platform

    origen = "DECA_DATA_DIR" if os.environ.get("DECA_DATA_DIR") else "autodeteccion"
    print(f"Sistema  = {platform.system()} ({os.name})")
    print(f"DATA_DIR = {DATA_DIR}  (via {origen})\n")
    for nombre, path in [
        ("code15", CODE15_DIR),
        ("samitrop", SAMITROP_DIR),
        ("ptbxl", PTBXL_DIR),
        ("challenge2021 raw", CHALLENGE2021_RAW_DIR),
        ("challenge2021 hdf5", CHALLENGE2021_HDF5),
        ("challenge2021 labels", CHALLENGE2021_LABELS_CSV),
        ("metadata.parquet", METADATA_PATH),
        ("fase2 hdf5", FASE2_HDF5),
        ("fase2 metadata", FASE2_METADATA_PATH),
    ]:
        print(f"  {'OK   ' if path.exists() else 'FALTA'} {nombre:<16} {path}")
