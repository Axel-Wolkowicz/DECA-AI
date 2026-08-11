"""Verifica la salida de Fase 2 y, opcionalmente, filtra del metadata las filas rotas.

Barre fase2_preprocessed.hdf5 buscando ventanas con valores no finitos (NaN o inf) y
cruza lo que encuentra contra fase2_metadata.parquet, que es lo que lee el entrenamiento.
Un solo NaN en un lote vuelve NaN la perdida, despues los gradientes y despues todos los
pesos de la red, sin recuperacion posible: el sintoma seria una perdida que explota sin
causa aparente, muy caro de diagnosticar a mitad de un entrenamiento.

Distingue dos casos, que no son lo mismo:
  - referenciada: el metadata apunta a esa fila. PELIGROSA, llega al entrenamiento.
  - huerfana: la fila esta en el HDF5 pero ninguna fila del metadata la referencia. Es
    inofensiva y es el estado esperado despues de un --fix (no se reescriben los 48 GB
    para borrar una fila; alcanza con que el metadata no la nombre).

Existe porque el arreglo del 2026-08-10 se hizo a mano sobre el parquet (ver FASES.md,
Fase 2): si algun dia se regenera el metadata, ese filtro hay que volver a aplicarlo, y
a mano no es reproducible. Con --fix esto lo reemplaza. Correrlo tambien despues de
cualquier corrida de preprocess.py, antes de copiar el HDF5 a la maquina de entrenamiento.

Uso:
    python src/verify_preprocessed.py            # solo reporta; sale con codigo 1 si hay problemas
    python src/verify_preprocessed.py --fix      # ademas reescribe el parquet sin las filas rotas
    python src/verify_preprocessed.py --limit N  # barre solo las primeras N filas (prueba rapida)
"""
import argparse
import sys
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FASE2_HDF5, FASE2_METADATA_PATH

# Cuantas ventanas se leen por vez. Con (2800, 12) float32 son ~134 KB por ventana, asi
# que 256 son ~34 MB por bloque: lee de a slabs contiguos en vez de fila por fila, que
# sobre el SSD externo por USB es lo que hace la diferencia.
BLOQUE = 256


def filas_no_finitas(tracings, limite: int | None = None) -> np.ndarray:
    """Indices (absolutos en el HDF5) de las ventanas con algun NaN o inf."""
    n = tracings.shape[0] if limite is None else min(limite, tracings.shape[0])
    malas = []
    for inicio in tqdm(range(0, n, BLOQUE), desc="barriendo", unit="bloque"):
        bloque = tracings[inicio : min(inicio + BLOQUE, n)]
        rotas = np.where(~np.isfinite(bloque).all(axis=(1, 2)))[0]
        malas.extend(inicio + rotas)
    return np.array(malas, dtype=np.int64)


def revisar_metadata(meta: pd.DataFrame, n_filas_hdf5: int) -> list[str]:
    """Chequeos baratos del parquet contra el HDF5, sin leer señal."""
    problemas = []
    indices = meta["row_index"]
    if indices.duplicated().any():
        problemas.append(f"{int(indices.duplicated().sum())} filas del metadata comparten row_index")
    fuera = indices[(indices < 0) | (indices >= n_filas_hdf5)]
    if len(fuera):
        problemas.append(f"{len(fuera)} filas del metadata apuntan fuera del HDF5 (0..{n_filas_hdf5 - 1})")
    return problemas


def ruta_backup():
    """Nombre del backup. Si ya existe uno (el del arreglo manual del 2026-08-10), se le
    agrega timestamp: pisar un backup previo seria justo lo que un backup evita."""
    backup = FASE2_METADATA_PATH.with_name("fase2_metadata_sin_filtrar.parquet")
    if backup.exists():
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = FASE2_METADATA_PATH.with_name(f"fase2_metadata_sin_filtrar_{sello}.parquet")
    return backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Reescribir el parquet sin las filas rotas (con backup)")
    parser.add_argument("--limit", type=int, default=None, help="Barrer solo las primeras N filas del HDF5")
    args = parser.parse_args()

    meta = pd.read_parquet(FASE2_METADATA_PATH)
    print(f"metadata: {len(meta)} filas ({FASE2_METADATA_PATH})")

    with h5py.File(FASE2_HDF5, "r") as f:
        tracings = f["tracings"]
        n_filas = tracings.shape[0]
        print(f"hdf5:     {n_filas} filas, shape por registro {tracings.shape[1:]} ({FASE2_HDF5})")

        n_barrido = n_filas if args.limit is None else min(args.limit, n_filas)
        problemas = revisar_metadata(meta, n_filas)
        malas = filas_no_finitas(tracings, args.limit)

    referenciadas = meta[meta["row_index"].isin(malas)]
    huerfanas = len(malas) - len(referenciadas)

    parcial = "" if n_barrido == n_filas else f"  (BARRIDO PARCIAL: --limit {args.limit})"
    print(f"\nFilas con valores no finitos: {len(malas)} de {n_barrido} barridas{parcial}")
    print(f"  huerfanas (ninguna fila del metadata las referencia): {huerfanas}  -- inofensivas")
    print(f"  referenciadas por el metadata:                        {len(referenciadas)}  -- llegan al entrenamiento")

    if len(referenciadas):
        columnas = [c for c in ("record_id", "dataset", "split", "chagas_label", "row_index") if c in referenciadas]
        print("\n" + referenciadas[columnas].to_string(index=False))

    for p in problemas:
        print(f"\nPROBLEMA: {p}")

    if not len(referenciadas) and not problemas:
        if n_barrido < n_filas:
            print(f"\nSin problemas en las {n_barrido} filas barridas, pero faltan {n_filas - n_barrido}: correr sin --limit para avalar el HDF5 entero.")
        else:
            print("\nOK: ninguna fila rota llega al entrenamiento.")
        return 0

    if not args.fix:
        print("\nCorrer con --fix para reescribir el parquet sin las filas rotas.")
        return 1

    if problemas:
        print("\nNo se toca el parquet: --fix solo filtra filas rotas, no arregla row_index invalidos.")
        return 1

    backup = ruta_backup()
    meta.to_parquet(backup, index=False)
    limpio = meta[~meta["row_index"].isin(malas)]
    limpio.to_parquet(FASE2_METADATA_PATH, index=False)
    print(f"\nBackup del metadata sin filtrar: {backup}")
    print(f"Metadata reescrito: {len(limpio)} filas ({len(meta) - len(limpio)} eliminadas) -> {FASE2_METADATA_PATH}")
    print("Las filas rotas siguen en el HDF5 pero ya nadie las referencia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
