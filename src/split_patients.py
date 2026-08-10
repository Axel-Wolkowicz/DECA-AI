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
import numpy as np
import pandas as pd

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def asignar_split(meta: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Devuelve una Series alineada con meta.index, con valores 'train'/'val'/'test'."""
    pacientes = (
        meta.groupby(["dataset", "patient_id"])["chagas_label"]
        .any()
        .reset_index()
        .rename(columns={"chagas_label": "label_paciente"})
    )
    pacientes["estrato"] = pacientes["dataset"] + "_" + pacientes["label_paciente"].astype(str)

    rng = np.random.default_rng(seed)
    asignaciones = []
    for _, grupo in pacientes.groupby("estrato"):
        n = len(grupo)
        orden = rng.permutation(n)
        n_train = round(n * RATIOS["train"])
        n_val = round(n * RATIOS["val"])
        etiquetas = np.empty(n, dtype=object)
        etiquetas[orden[:n_train]] = "train"
        etiquetas[orden[n_train : n_train + n_val]] = "val"
        etiquetas[orden[n_train + n_val :]] = "test"
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
