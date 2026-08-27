"""Asigna train/val/test por paciente, no por examen.

Por que por paciente: un mismo paciente puede tener varios examenes (code15: hasta 38;
ptbxl: hasta varios tambien, 2.111 pacientes con mas de un registro). Si dos examenes
del mismo paciente caen en splits distintos, el modelo puede aprender a reconocer a ESE
paciente (edad, morfologia de base, ruido de adquisicion propio) en vez del patron
clinico general, e infla la metrica de test sin que eso generalice a pacientes nuevos.
Por eso se agrupa por (dataset, patient_id) y el grupo entero -- todos sus examenes --
va al mismo split. No se descarta ningun registro por tener paciente repetido: se decide
en bloque, por paciente, a que split va cada uno.

Estratificado por (dataset, label a nivel paciente) para no desbalancear los splits: sin
esto, un split al azar podria dejar casi todos los positivos fuertes de SaMi-Trop de un
lado.

Uso: from split_patients import asignar_split
"""
import hashlib

import numpy as np
import pandas as pd

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _rng_estrato(seed: int, estrato: str) -> np.random.Generator:
    """Un generador propio por estrato, derivado de (seed, nombre del estrato).

    **Por que no un rng compartido** (que es como estaba y se midio el 2026-08-27): con un
    solo `rng` consumido en el loop, los estratos se sirven en orden alfabetico y cada uno
    corre el estado del generador para los siguientes. Agregar un dataset nuevo cuyo nombre
    ordene antes -- p.ej. `challenge2021_False` va antes que `code15_False` -- desplaza el
    estado de TODOS los estratos posteriores y reasigna pacientes que no tenian por que
    moverse. Medido: 45,9% de los pacientes cambiaban de split y 26.617 pasaban de test a
    train, o sea fuga directa del conjunto de test.

    Derivando la semilla del nombre del estrato, cada uno es independiente: sumar estratos
    nuevos no puede tocar a los viejos. Se usa sha256 y no `hash()` porque el hash de
    strings de Python esta aleatorizado por proceso (PYTHONHASHSEED) y no seria reproducible.
    """
    h = int.from_bytes(hashlib.sha256(estrato.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng([seed, h])


def asignar_split(
    meta: pd.DataFrame, seed: int = 42, congelado: pd.DataFrame | None = None
) -> pd.Series:
    """Devuelve una Series alineada con meta.index, con valores 'train'/'val'/'test'.

    `congelado` es un DataFrame (dataset, patient_id, split) con asignaciones ya tomadas:
    esos pacientes conservan su split exacto y solo se sortean los que no aparecen ahi. Es
    lo que permite sumar un dataset sin invalidar las metricas de todos los modelos ya
    entrenados. `config.SPLIT_CONGELADO_PATH` guarda la tabla canonica, verificada contra
    el split que realmente uso la Fase 2 (0 discrepancias sobre 252.227 pacientes).
    """
    pacientes = (
        meta.groupby(["dataset", "patient_id"])["chagas_label"]
        .any()
        .reset_index()
        .rename(columns={"chagas_label": "label_paciente"})
    )
    pacientes["estrato"] = pacientes["dataset"] + "_" + pacientes["label_paciente"].astype(str)

    previo = {}
    if congelado is not None and len(congelado):
        faltan = {"dataset", "patient_id", "split"} - set(congelado.columns)
        if faltan:
            raise ValueError(f"al split congelado le faltan columnas: {sorted(faltan)}")
        previo = {
            (d, p): sp
            for d, p, sp in congelado[["dataset", "patient_id", "split"]].itertuples(index=False)
        }

    asignaciones = []
    for estrato, grupo in pacientes.groupby("estrato"):
        grupo = grupo.reset_index(drop=True)
        claves = list(zip(grupo["dataset"], grupo["patient_id"]))
        fijos = np.array([previo.get(k) for k in claves], dtype=object)
        libres = np.flatnonzero(pd.isna(fijos))

        etiquetas = fijos.copy()
        n = len(libres)
        if n:
            # Solo se sortean los pacientes sin asignacion previa. Los ratios se aplican
            # sobre ese remanente: los congelados ya aportan su propia proporcion.
            rng = _rng_estrato(seed, estrato)
            orden = rng.permutation(n)
            n_train = round(n * RATIOS["train"])
            n_val = round(n * RATIOS["val"])
            nuevas = np.empty(n, dtype=object)
            nuevas[orden[:n_train]] = "train"
            nuevas[orden[n_train : n_train + n_val]] = "val"
            nuevas[orden[n_train + n_val :]] = "test"
            etiquetas[libres] = nuevas
        asignaciones.append(grupo.assign(split=etiquetas))

    pacientes_split = pd.concat(asignaciones, ignore_index=True)[["dataset", "patient_id", "split"]]

    meta_con_split = meta.merge(pacientes_split, on=["dataset", "patient_id"], how="left", validate="many_to_one")
    assert meta_con_split["split"].notna().all(), "quedaron registros sin split asignado"
    return meta_con_split.set_index(meta.index)["split"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from eda_utils import cargar_metadata

    meta = cargar_metadata()
    meta["split"] = asignar_split(meta)

    print("Registros por split y dataset:")
    print(meta.groupby(["split", "dataset"]).size().unstack(fill_value=0))
    print("\nPacientes por split y dataset:")
    print(meta.groupby(["split", "dataset"])["patient_id"].nunique().unstack(fill_value=0))
    print("\n% positivos por split (a nivel registro):")
    print(meta.groupby("split")["chagas_label"].mean().mul(100).round(2))
