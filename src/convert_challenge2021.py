"""Convierte Challenge 2021 (WFDB, carpeta training/) a un HDF5 unico + labels de patron.

A diferencia de PTB-XL, este dataset NO trae etiqueta de Chagas. Trae diagnostico
SNOMED-CT por registro (comentario "# Dx:" del header WFDB), que se mapea a los patrones
de ECG objetivo del ROADMAP (BRD, HBAI, extrasistoles ventriculares, onda Q anormal, mala
progresion de onda R) via el subconjunto relevante de dx_mapping_scored.csv (repo
physionetchallenges/evaluation-2021).

Correccion sobre lo escrito en FASES.md/ROADMAP.md el 2026-08-26: la sesion de ese dia
listaba INCART (74 registros) y PTB/PTB-XL (21.837) entre las fuentes de Challenge 2021.
El manifiesto real de la version 1.0.3 (SHA256SUMS.txt) NO los trae bajo training/: las
unicas 5 fuentes presentes son chapman_shaoxing, cpsc_2018, cpsc_2018_extra, georgia y
ningbo (125.758 archivos = 2 por registro, .hea+.mat). No hace falta excluir INCART a mano
ni deduplicar contra el PTB-XL que ya tenemos -- ninguno de los dos esta en este dataset.

Forma de la señal: se usa la misma convencion que ptbxl.hdf5 (500 Hz, 5000 muestras = 10s),
no la nativa de cada registro (que varia: la mayoria son 10s pero CPSC/CPSC-Extra van de
~6 a ~60s). Los registros mas largos se recortan CENTRADOS a 5000 muestras y los mas
cortos se rellenan con cero al final. Es un recorte sin perdida para el pipeline tal como
esta: Fase 2 (preprocess.py) tambien recorta centrado (a 7.0s) y descarta el padding de
ceros antes de usar la señal, asi que el centro que ve Fase 2 es el mismo center que
tendria si se le dieran los N muestras nativos completos. Ver docstring de encajar_ventana.

Este dataset NO se integra todavia a metadata.parquet/build_metadata.py ni a las cabezas
del modelo (model.py/dataset.py/train.py): sumarlo ahi implica enmascarar la cabeza de
Chagas para estos registros (mismo mecanismo que ya usa RBBB) y verificar que no reabra el
atajo de fuente -- decision de arquitectura que FASES.md marca como pendiente y a resolver
midiendo, no a priori. Este script solo dexa la señal y los labels de patron en un formato
limpio, listo para ese paso futuro.

Uso: python src/convert_challenge2021.py
"""
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from config import CHALLENGE2021_DIR, CHALLENGE2021_HDF5, CHALLENGE2021_LABELS_CSV

FREQ = 500
N_SAMPLES = 5000  # 10s a 500 Hz -- misma convencion que ptbxl.hdf5
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

FUENTES = ["chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"]

# Subconjunto de dx_mapping_scored.csv relevante a los patrones objetivo del ROADMAP.
# SNOMED-CT -> patron. Equivalencias segun la columna "Notes" del csv oficial (mismo
# hallazgo clinico, dos codigos): CRBBB/RBBB son la misma entidad puntuada juntas, PVC/VPB
# tambien. Fuente: https://github.com/physionetchallenges/evaluation-2021 dx_mapping_scored.csv
SNOMED_A_PATRON = {
    "713427006": "brd",   # complete right bundle branch block (CRBBB)
    "59118001": "brd",    # right bundle branch block (RBBB) -- misma entidad que CRBBB
    "713426002": "brd",   # incomplete right bundle branch block (IRBBB)
    "445118002": "hbai",  # left anterior fascicular block (LAnFB)
    "427172004": "pvc",   # premature ventricular contractions (PVC)
    "17338001": "pvc",    # ventricular premature beats (VPB) -- misma entidad que PVC
    "164884008": "pvc",   # ventricular ectopics (VEB) -- ver "Auditoria" abajo
    "11157007": "pvc",    # ventricular bigeminy -- un patron DE extrasistoles
    "251180001": "pvc",   # ventricular trigeminy -- idem
    "164917005": "qab",   # qwave abnormal (QAb)
    "365413008": "prwp",  # poor R wave progression (PRWP)
}

# Auditoria completa contra dx_mapping_scored.csv + dx_mapping_unscored.csv (2026-08-27):
# se cruzaron los 118 codigos SNOMED presentes en el corpus contra su nombre oficial. BRD,
# HBAI, QAb y PRWP quedaron COMPLETOS -- ningun codigo faltante ni mal asignado. El unico
# patron con huecos era extrasistoles, y el mas grande era `164884008` (741 apariciones,
# 700 de ellas en cpsc_2018, que es exactamente su clase "PVC" oficial: por eso esa fuente
# daba pvc=0 antes de este arreglo).
#
# **Codigos deliberadamente NO mapeados a pvc**, para que no los "arregle" alguien despues:
#   81898007  ventricular escape rhythm  -- mecanismo OPUESTO: el ventriculo suple un fallo
#   75532003  ventricular escape beat    -- del marcapasos sinusal, no se adelanta
#   63593006  supraventricular premature beats -- supraventricular, no ventricular
#   251173003 atrial bigeminy            -- auricular, no ventricular
# Bigeminia y trigeminia SI entran porque son, por definicion, secuencias de extrasistoles
# ventriculares -- y porque es lo que este repo ya hace para PTB-XL (FASES.md, hallazgo 5,
# donde el patron se agrupa como PVC/BIGU/TRIGU/PRC(S)). Deja los dos datasets consistentes.
PATRONES = ["brd", "hbai", "pvc", "qab", "prwp"]


def parse_comments(comments: list[str]) -> dict[str, str]:
    """Header WFDB trae 'Age: 85' / 'Sex: Male' / 'Dx: 164889003,59118001' como comentarios."""
    out = {}
    for c in comments:
        if ":" not in c:
            continue
        k, v = c.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def encajar_ventana(señal: np.ndarray) -> np.ndarray:
    """Deja la señal en (N_SAMPLES, 12): recorte centrado si sobra, cero al final si falta.

    Recortar centrado (no desde el inicio) es lo que hace que este paso no pierda
    informacion relevante: Fase 2 tambien recorta centrado (a 7.0s) sobre lo que reciba,
    asi que el punto medio de un recorte centrado de 5000 coincide con el punto medio del
    registro original completo, sea cual sea su duracion nativa.
    """
    n = señal.shape[0]
    if n == N_SAMPLES:
        return señal
    if n > N_SAMPLES:
        inicio = (n - N_SAMPLES) // 2
        return señal[inicio : inicio + N_SAMPLES]
    out = np.zeros((N_SAMPLES, señal.shape[1]), dtype=señal.dtype)
    out[:n] = señal
    return out


def listar_registros() -> list[tuple[str, "Path"]]:
    """(fuente, path_sin_extension) para cada .hea bajo training/<fuente>/*/*.hea."""
    registros = []
    for fuente in FUENTES:
        base = CHALLENGE2021_DIR / "training" / fuente
        if not base.is_dir():
            continue
        for hea in sorted(base.glob("*/*.hea")):
            registros.append((fuente, hea.with_suffix("")))
    return registros


def main():
    if CHALLENGE2021_HDF5.exists():
        sys.exit(f"{CHALLENGE2021_HDF5} ya existe. Borrarlo a mano si se quiere regenerar.")

    registros = listar_registros()
    n = len(registros)
    if n == 0:
        sys.exit(f"No se encontraron .hea bajo {CHALLENGE2021_DIR / 'training'}")
    print(f"Challenge 2021: {n} registros .hea encontrados a convertir")

    tmp_path = CHALLENGE2021_HDF5.with_suffix(".hdf5.tmp")
    filas = []
    saltados = Counter()
    con_nan = []

    with h5py.File(tmp_path, "w") as f:
        tracings = f.create_dataset(
            "tracings",
            shape=(n, N_SAMPLES, 12),
            maxshape=(n, N_SAMPLES, 12),
            dtype="float32",
            chunks=(1, N_SAMPLES, 12),
        )
        f.attrs["frecuencia"] = FREQ
        f.attrs["duracion"] = N_SAMPLES / FREQ
        f.attrs["leads"] = LEADS
        f.attrs["fuente"] = "PhysioNet/CinC Challenge 2021 v1.0.3, carpeta training/"

        idx = 0
        for fuente, path in tqdm(registros, unit="reg"):
            try:
                rec = wfdb.rdrecord(str(path))
            except Exception as e:
                saltados[f"{fuente}: error de lectura ({type(e).__name__})"] += 1
                continue

            if rec.fs != FREQ:
                saltados[f"{fuente}: frecuencia inesperada ({rec.fs} Hz)"] += 1
                continue

            nombres = [s.upper() for s in rec.sig_name]
            if nombres == LEADS:
                señal = rec.p_signal
            else:
                try:
                    orden = [nombres.index(lead) for lead in LEADS]
                    señal = rec.p_signal[:, orden]
                except ValueError:
                    saltados[f"{fuente}: derivaciones inesperadas {rec.sig_name}"] += 1
                    continue

            if señal.shape[1] != 12:
                saltados[f"{fuente}: shape inesperado {señal.shape}"] += 1
                continue

            duracion_original = señal.shape[0] / FREQ
            if np.isnan(señal).any():
                con_nan.append(f"{fuente}/{path.name}")

            tracings[idx] = encajar_ventana(señal.astype(np.float32))

            comentarios = parse_comments(rec.comments)
            edad_raw = comentarios.get("age", "")
            sexo_raw = comentarios.get("sex", "").strip().lower()
            dx_raw = comentarios.get("dx", "")
            codigos = [c.strip() for c in dx_raw.split(",") if c.strip() and c.strip().lower() != "unknown"]

            try:
                edad = float(edad_raw)
            except ValueError:
                edad = np.nan

            if sexo_raw.startswith("m"):
                sexo = "M"
            elif sexo_raw.startswith("f"):
                sexo = "F"
            else:
                sexo = None

            patrones = {p: False for p in PATRONES}
            for c in codigos:
                p = SNOMED_A_PATRON.get(c)
                if p:
                    patrones[p] = True

            filas.append({
                "record_id": f"{fuente}/{path.name}",
                "fuente": fuente,
                "patient_id": f"{fuente}/{path.name}",  # sin id de paciente en la fuente; proxy
                "row_index": idx,
                "edad": edad,
                "sexo": sexo,
                "frecuencia": FREQ,
                "duracion_original": duracion_original,
                "dx_codigos": ";".join(codigos),
                "con_nan": bool(np.isnan(señal).any()),
                "brd_y_hbai": patrones["brd"] and patrones["hbai"],  # patron clasico de Chagas
                **patrones,
            })
            idx += 1

        tracings.resize((idx, N_SAMPLES, 12))

    tmp_path.replace(CHALLENGE2021_HDF5)

    labels = pd.DataFrame(filas)
    labels.to_csv(CHALLENGE2021_LABELS_CSV, index=False)

    print(f"\n{idx}/{n} registros convertidos -> {CHALLENGE2021_HDF5}")
    print(f"Labels -> {CHALLENGE2021_LABELS_CSV}")

    if saltados:
        print(f"\nSalteados ({sum(saltados.values())} en total):")
        for motivo, cantidad in sorted(saltados.items()):
            print(f"  {cantidad:>6}  {motivo}")

    if con_nan:
        print(f"\n{len(con_nan)} registros con NaN en la señal (se guardan igual; Fase 2 los descarta solo)")

    print("\nPor fuente:")
    print(labels.groupby("fuente").size())

    print("\nPositivos por patron (y % sobre el total convertido):")
    for p in PATRONES + ["brd_y_hbai"]:
        pos = int(labels[p].sum())
        print(f"  {p:<10} {pos:>7}  ({pos / idx * 100:.2f}%)")


if __name__ == "__main__":
    main()
