"""Fase 4: Dataset de PyTorch sobre la salida de Fase 2 (fase2_preprocessed.hdf5).

Tres cosas que este modulo resuelve y que no son obvias:

1. **El label de RBBB no esta en fase2_metadata.parquet.** Vive en code15/exams.csv, que
   es la unica fuente que lo trae (SaMi-Trop solo aporta normal_ecg, PTB-XL usa otra
   taxonomia). Se hace el merge aca por record_id y los registros sin dato quedan con
   mascara 0: la loss de esa cabeza los ignora, no los cuenta como negativos. Un RBBB
   ausente por no estar anotado no es un "no tiene RBBB" (ver Fase 3, decision 1).

2. **PTB-XL se excluye del entrenamiento por completo** (decision del 2026-08-12,
   FASES.md Fase 4). Aporta 15.286 negativos contra los ~232.000 que ya da CODE-15%, y a
   cambio mete la unica pista de fuente que el preprocesado de Fase 2 no logro borrar: su
   contenido de alta frecuencia lo separa de CODE-15% con AUC 0,880 aun despues del
   z-score. Como PTB-XL es 100% negativo y SaMi-Trop 100% positivo, aprender esa pista es
   equivalente a aprender la etiqueta. Sigue cargandose en val/test: ahi es la arena C
   (especificidad cruzada de poblacion), y justamente porque el modelo nunca la vio
   entrenando, esa arena mide generalizacion real. El flag --con-ptbxl corre la ablacion
   inversa.

3. **h5py no se puede compartir entre procesos.** Con num_workers>0 cada worker abre su
   propio handle la primera vez que pide un registro (apertura perezosa en __getitem__).
   Abrirlo en __init__ hace que el handle se copie al hacer fork/spawn y da lecturas
   corruptas o cuelgues.

Sobre PESO_STRONG=1,0 (ponderacion por confianza, Fase 3 decision 4). SaMi-Trop es 0,45%
de los registros de train, pero eso NO es lo que ve el optimizador: como todos sus
registros son positivos, `pos_weight` (41x) ya le corrige el submuestreo y con peso 1,0
ya se lleva el 19,1% del gradiente de la clase positiva. El eje que `peso_strong` mueve es
otro: cuanto vale una etiqueta serologica contra una autorreportada. Medido sobre el train
real (238.027 registros, 4.574 positivos autorreportados + 1.083 serologicos):

    peso_strong   pos_weight   masa+ de SaMi-Trop   exposicion efectiva   AUPRC arena A
        1,0          41,1            19,1%                  41x            0,1755  <- default
        3,0          29,7            41,5%                  89x            0,1500
        5,0          23,3            54,2%                 116x            0,1460

Razonamiento teorico original (docstring previo): subir peso_strong dejaria a la
serologia como co-protagonista de la señal positiva sin dominarla del todo, y por eso
se eligio 3,0 como default inicial. El barrido empirico (--peso-strong, FASES.md Fase 4,
2026-08-13) lo contradijo: AUPRC de arena A cae monotonicamente al subir peso_strong, y
el tamaño del cruce del atajo de fuente tambien crece con el (0,0021 -> 0,0059 -> 0,0077).
1,0 gana en las dos metricas que importan, asi que queda como default. Efecto lateral a
tener presente: subir peso_strong BAJA pos_weight (41 -> 29,7), o sea que de paso les
baja el peso a los positivos de CODE-15% — parte de por que empeora arena A.
"""
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CODE15_EXAMS_CSV, FASE2_HDF5, FASE2_METADATA_PATH

# Pesos por tier de confianza de la etiqueta (columna `confianza` de metadata).
# El 1.0 de `strong` esta argumentado con numeros en el docstring del modulo.
PESO_STRONG = 1.0
PESOS_CONFIANZA = {"weak": 1.0, "strong": PESO_STRONG, "negativo-presunto": 1.0}


def cargar_metadata_fase4(peso_strong: float | None = None) -> pd.DataFrame:
    """fase2_metadata.parquet + label de RBBB (con mascara) + peso por confianza.

    Columnas agregadas:
        rbbb_label  float32, 0.0/1.0; 0.0 tambien donde no hay dato (lo tapa la mascara)
        rbbb_mask   float32, 1.0 si el registro tiene label de RBBB, 0.0 si no
        peso        float32, peso de la muestra en la loss de Chagas
    """
    meta = pd.read_parquet(FASE2_METADATA_PATH)

    rbbb = pd.read_csv(CODE15_EXAMS_CSV, usecols=["exam_id", "RBBB"])
    rbbb["record_id"] = rbbb["exam_id"].astype(str)
    # La clave lleva `dataset`, no solo `record_id`: los record_id NO son unicos entre
    # datasets. El de PTB-XL es su ecg_id (1..21.837) y esos mismos numeros existen como
    # exam_id en CODE-15%, asi que mergear solo por record_id le pega label de RBBB
    # ajeno a 3.100 registros de PTB-XL -- en silencio, y encima en la arena C.
    rbbb["dataset"] = "code15"
    meta = meta.merge(rbbb[["dataset", "record_id", "RBBB"]], on=["dataset", "record_id"], how="left")

    meta["rbbb_mask"] = meta["RBBB"].notna().astype(np.float32)
    meta["rbbb_label"] = meta["RBBB"].fillna(False).astype(bool).astype(np.float32)
    meta = meta.drop(columns=["RBBB"])

    pesos = dict(PESOS_CONFIANZA)
    if peso_strong is not None:
        pesos["strong"] = peso_strong
    meta["peso"] = meta["confianza"].map(pesos).astype(np.float32)
    if meta["peso"].isna().any():
        faltantes = sorted(meta.loc[meta["peso"].isna(), "confianza"].unique())
        raise ValueError(f"tier de confianza sin peso definido en PESOS_CONFIANZA: {faltantes}")

    return meta


def filtrar_split(
    meta: pd.DataFrame,
    split: str,
    con_ptbxl: bool = False,
    limite: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Subconjunto de un split. `con_ptbxl` solo aplica a train (val/test siempre lo llevan:
    es la arena C). `limite` toma una muestra ALEATORIA, no las primeras N filas: el
    parquet viene ordenado por (dataset, source_file) desde preprocess.py, asi que las
    primeras N serian todas de code15/exams_part0 y ademas casi todas negativas."""
    sub = meta[meta["split"] == split]
    if split == "train" and not con_ptbxl:
        sub = sub[sub["dataset"] != "ptbxl"]
    if limite is not None and limite < len(sub):
        sub = sub.sample(n=limite, random_state=seed)
    return sub.reset_index(drop=True)


EDAD_TOPE = 90.0   # ver normalizar_demograficos
EDAD_CENTRO = 50.0
EDAD_ESCALA = 25.0


def normalizar_demograficos(meta: pd.DataFrame) -> np.ndarray:
    """(N, 2) float32 con [edad normalizada, es_hombre]. Sin faltantes en los 3 datasets.

    **El tope de edad no es cosmetico.** PTB-XL codifica "mayor de 89" como `edad=300`
    (convencion de anonimizacion de la fuente): son 293 registros que, sin recortar,
    entran a la red como personas de 300 anios y dominan cualquier normalizacion. Sacando
    los 300, el maximo de PTB-XL es exactamente 89. Se recorta a 90 porque es lo que el
    centinela realmente significa (89+), no un valor inventado.

    La escala es fija y no depende de los datos (centro 50, escala 25, ~la media y 1,3
    desvios de CODE-15%) para que train/val/test y cualquier corrida futura vean la misma
    transformacion -- normalizar con estadisticos del split traeria fuga.
    """
    edad = meta["edad"].to_numpy(dtype=np.float32)
    edad = np.clip(edad, 0.0, EDAD_TOPE)
    edad = (edad - EDAD_CENTRO) / EDAD_ESCALA
    es_hombre = (meta["sexo"].to_numpy() == "M").astype(np.float32)
    return np.stack([edad, es_hombre], axis=1).astype(np.float32)


class ECGDataset(Dataset):
    """Devuelve (señal (12, 2800), chagas, rbbb, rbbb_mask, peso, demo).

    `demo` es (2,) = [edad normalizada, es_hombre]. Se devuelve SIEMPRE, aunque el modelo
    corra sin demograficos (`--con-demograficos` apagado): mantener la tupla de tamanio
    fijo evita que el desempaquetado del loop dependa de un flag.

    La señal se transpone de (2800, 12) a (12, 2800) porque Conv1d espera (canales,
    tiempo). El orden de las filas es el del DataFrame que se le pasa, asi que con
    shuffle=False las predicciones salen alineadas con `meta` por posicion -- de eso
    depende la evaluacion por arenas en evaluar.py.
    """

    def __init__(self, meta: pd.DataFrame, hdf5_path=FASE2_HDF5):
        self.hdf5_path = str(hdf5_path)
        self.meta = meta.reset_index(drop=True)
        self.row_index = self.meta["row_index"].to_numpy(dtype=np.int64)
        self.chagas = self.meta["chagas_label"].to_numpy(dtype=np.float32)
        self.rbbb = self.meta["rbbb_label"].to_numpy(dtype=np.float32)
        self.rbbb_mask = self.meta["rbbb_mask"].to_numpy(dtype=np.float32)
        self.peso = self.meta["peso"].to_numpy(dtype=np.float32)
        self.demo = normalizar_demograficos(self.meta)
        self._h5 = None  # apertura perezosa: ver punto 3 del docstring del modulo

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        if self._h5 is None:
            # locking=False: en Linux, HDF5 pide un lock de archivo al abrir y en exFAT
            # (que es como esta formateado el SSD) eso falla con "unable to lock file",
            # aun para lectura y aun con un solo proceso. En Windows no pasa, asi que el
            # sintoma solo aparece al mover el entrenamiento a la maquina con la 4090.
            # Es seguro porque el archivo se abre de solo lectura y nadie lo escribe
            # mientras se entrena.
            self._h5 = h5py.File(self.hdf5_path, "r", locking=False)
        señal = self._h5["tracings"][self.row_index[i]]  # (2800, 12) float32
        x = torch.from_numpy(np.ascontiguousarray(señal.T))  # (12, 2800)
        return (
            x,
            torch.tensor(self.chagas[i]),
            torch.tensor(self.rbbb[i]),
            torch.tensor(self.rbbb_mask[i]),
            torch.tensor(self.peso[i]),
            torch.from_numpy(self.demo[i]),
        )

    def __getstate__(self):
        # El handle de h5py no es picklable; los workers lo reabren solos.
        estado = self.__dict__.copy()
        estado["_h5"] = None
        return estado


def pos_weight_chagas(meta: pd.DataFrame) -> float:
    """n_negativos/n_positivos ponderado por `peso`, para BCEWithLogitsLoss.

    Es la alternativa al oversampling decidida el 2026-08-12 (FASES.md Fase 4): reescala
    el gradiente de los positivos en vez de repetir sus grabaciones, asi que no acumula
    exposiciones a un puñado de registros.
    """
    y = meta["chagas_label"].to_numpy(dtype=np.float32)
    w = meta["peso"].to_numpy(dtype=np.float32)
    pos, neg = float((w * y).sum()), float((w * (1 - y)).sum())
    if pos == 0:
        raise ValueError("no hay positivos en el conjunto: no se puede calcular pos_weight")
    return neg / pos


def pos_weight_rbbb(meta: pd.DataFrame) -> float:
    """Igual que el de Chagas pero solo sobre los registros que tienen label de RBBB."""
    sub = meta[meta["rbbb_mask"] > 0]
    y = sub["rbbb_label"].to_numpy(dtype=np.float32)
    pos, neg = float(y.sum()), float((1 - y).sum())
    if pos == 0:
        raise ValueError("no hay positivos de RBBB en el conjunto")
    return neg / pos


if __name__ == "__main__":
    meta = cargar_metadata_fase4()
    print(f"{len(meta)} registros en fase2_metadata + label de RBBB\n")

    print("Cobertura del label de RBBB (mascara=1) por dataset:")
    cob = meta.groupby("dataset").agg(
        n=("record_id", "count"),
        con_rbbb=("rbbb_mask", "sum"),
        positivos_rbbb=("rbbb_label", "sum"),
    )
    cob["% con label"] = (cob["con_rbbb"] / cob["n"] * 100).round(1)
    print(cob, "\n")

    for split in ("train", "val", "test"):
        sub = filtrar_split(meta, split)
        print(f"{split:<6} {len(sub):>7} registros  "
              f"({', '.join(f'{d}={n}' for d, n in sub['dataset'].value_counts().items())})")

    train = filtrar_split(meta, "train")
    print(f"\npos_weight chagas (train, sin ptbxl) = {pos_weight_chagas(train):.1f}")
    print(f"pos_weight rbbb   (train, sin ptbxl) = {pos_weight_rbbb(train):.1f}")
