"""Valida las cabezas de patron contra anotacion de cardiologo en pacientes CHAGASICOS.

    python src/validar_patrones_samitrop.py --checkpoint D:/DECA-datasets/modelos/patrones-lr8/mejor.pt

**Por que existe.** Las cabezas de patron (`hbai`, `extra`, `zona`) y la de RBBB se entrenan
con PTB-XL y Challenge 2021 -- poblaciones NO chagasicas (alemanes de los 90 y una mezcla
internacional) -- y se evaluan en la misma val. Nunca se midio si transfieren a pacientes con
Chagas, que es donde el producto las va a usar. Si el producto es un priorizador que le dice
a un medico "testea a este porque tiene BRD", esa cabeza tiene que estar validada en la
poblacion donde se va a usar, no solo en la que la entreno.

**El dato.** `samitrop-lvsd/` (repositorio publico del grupo SaMi-Trop, ver FASES.md sesion
del 2026-09-10): 1.304 pacientes seropositivos confirmados, con ECG y las anormalidades
anotadas por cardiologo segun Minnesota Code, ademas de fraccion de eyeccion por
ecocardiograma.

**Que se valida y que NO.**
    cabeza rbbb  <- V20 (BRD completo/intermitente)   404 casos, 31,0%
    cabeza extra <- V24 (extrasistoles ventriculares)  33 casos,  2,5%
    cabeza zona  <- V18 (onda Q anormal)              156 casos, 12,0%
    cabeza hbai  <- NADA. **V21 ("BRD + HBAI") esta rota** -- da 9 casos (0,7%) contra los
                    223 (11,5%) publicados para la cohorte, mientras TODAS las demas columnas
                    coinciden con lo publicado dentro de 2,5 puntos. V20 (31,0%) coincide con
                    el BRD *total* publicado (31,3%), no con el BRD aislado (19,8%), asi que
                    V20 es "todo BRD" y V21 no es una marca usable de BRD+HBAI. Ver FASES.md.

**Se reporta AUC ademas de AUPRC, y el AUC es el que compara.** El AUPRC depende de la
prevalencia, y la de cada patron en esta cohorte no es la de nuestra val, asi que comparar
AUPRC entre las dos poblaciones mide en parte la diferencia de prevalencia y no la de
rendimiento. El AUC no tiene ese problema.

**Preprocesado: se importa `procesar_registro` de preprocess.py, no se reimplementa.** Asi
cualquier diferencia en el resultado no puede venir de una deriva del preprocesado. Lo unico
que este modulo hace antes de llamarlo es reordenar las derivaciones (la fuente trae
DI,DII,DIII,AVL,AVF,AVR,... y la convencion del proyecto es I,II,III,aVR,aVL,aVF,...) y pasar
300 Hz como frecuencia nativa. La resolucion del equipo (3,9 o 5,0 uV/LSB segun el registro)
NO se corrige a proposito: el z-score por registro y por derivacion la cancela.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from model import ResNet1D
from preprocess import procesar_registro

FREQ_NATIVA = 300
# Orden de la fuente -> convencion del proyecto (I, II, III, aVR, aVL, aVF, V1-V6).
# La fuente trae AVL y AVF ANTES que AVR; leerlas en el orden del archivo le entrega al
# modelo tres derivaciones permutadas, que es un error silencioso: la forma del tensor es
# correcta y el resultado solo sale "un poco peor".
LEADS_CANONICO = ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

# cabeza del modelo -> (columna de anotacion, nombre clinico)
VALIDACIONES = {
    "rbbb": ("V20", "BRD completo/intermitente"),
    "extra": ("V24", "extrasistoles ventriculares"),
    "zona": ("V18", "onda Q anormal"),
}
PATRONES_ORDEN = ("hbai", "extra", "zona")  # el orden de la cabeza, ver dataset.PATRONES


def cargar_trazados(trace_csv: str, ids_validos: set[str]) -> tuple[np.ndarray, list[str]]:
    """(N, 2800, 12) normalizado + la lista de id_exam en el mismo orden.

    **Cada examen tiene varios "registros" y NO todos traen las 12 derivaciones.** El equipo
    graba por grupos (el patron clasico de un ECG de 12 derivaciones), asi que en 903 de los
    1.638 registros nº1 algunas derivaciones vienen con un solo valor -- vacias de hecho -- y
    el resto con 3.360 muestras. Quedarse con el registro 1 a ciegas descarta el 55% del
    conjunto.

    Se toma **el primer registro que trae las 12 derivaciones con el mismo largo y al menos
    7 s**. Eso rescata 1.061 de los 1.304 examenes etiquetados (81%).

    **Lo que NO se hace, a proposito: completar las derivaciones faltantes de un registro con
    las de otro.** Uniendo registros se llegaria a 1.395 examenes, pero las derivaciones de
    registros distintos son tramos de tiempo distintos -- no son simultaneas. Todo el
    entrenamiento (CODE-15%, PTB-XL) es 12 derivaciones simultaneas, y un registro cosido
    seria una señal que el modelo nunca vio, con latidos que no se corresponden entre
    derivaciones. Los 243 examenes sin ningun registro completo se pierden; es el precio de
    no inventar la entrada.
    """
    df = pd.read_csv(trace_csv, sep=";", dtype=str)
    df = df[df["id_exam"].isin(ids_validos)]
    df = df.sort_values(["id_exam", "register_num"], key=lambda c: pd.to_numeric(c, errors="coerce"))

    señales, ids, descartes = [], [], {}
    for id_exam, grupo in df.groupby("id_exam", sort=False):
        elegido = None
        for _, fila in grupo.iterrows():
            if fila[LEADS_CANONICO].isna().any():
                continue
            derivaciones = [np.fromstring(fila[L], sep=",", dtype=np.float32)
                            for L in LEADS_CANONICO]
            largos = {len(d) for d in derivaciones}
            # Un unico largo compartido = las 12 se grabaron juntas. Distintos largos
            # significa que a este registro le faltan derivaciones (vienen con un solo valor).
            if len(largos) != 1 or largos.pop() < FREQ_NATIVA * 7.0:
                continue
            elegido = np.stack(derivaciones, axis=1)
            break

        if elegido is None:
            descartes["sin registro completo de 12 derivaciones"] = (
                descartes.get("sin registro completo de 12 derivaciones", 0) + 1)
            continue

        ventana, motivo = procesar_registro(elegido, FREQ_NATIVA)
        if ventana is None:
            descartes[motivo] = descartes.get(motivo, 0) + 1
            continue
        señales.append(ventana)
        ids.append(id_exam)

    if descartes:
        print("  descartes:", ", ".join(f"{k}={v}" for k, v in sorted(descartes.items())))
    return np.stack(señales), ids


@torch.no_grad()
def predecir(modelo, x: np.ndarray, device, batch: int = 128) -> dict[str, np.ndarray]:
    modelo.eval()
    chagas, rbbb, patrones = [], [], []
    for i in range(0, len(x), batch):
        lote = torch.from_numpy(x[i : i + batch].transpose(0, 2, 1)).to(device)  # (B,12,2800)
        demo = torch.zeros((len(lote), 2), device=device)
        lc, lr, lp = modelo(lote, demo)
        chagas.append(torch.sigmoid(lc.float()).cpu().numpy())
        rbbb.append(torch.sigmoid(lr.float()).cpu().numpy())
        patrones.append(torch.sigmoid(lp.float()).cpu().numpy())
    out = {"chagas": np.concatenate(chagas), "rbbb": np.concatenate(rbbb)}
    pat = np.concatenate(patrones)
    for j, nombre in enumerate(PATRONES_ORDEN):
        if j < pat.shape[1]:
            out[nombre] = pat[:, j]
    return out


def main():
    p = argparse.ArgumentParser(description="Valida las cabezas de patron en pacientes chagasicos")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dir", default="D:/DECA-datasets/samitrop-lvsd/x",
                   help="carpeta con los dos CSV extraidos del repositorio")
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    base = args.dir.rstrip("/")
    labels_csv = f"{base}/AI-ECG-LVSD-Chagas_Sami-Trop/AI-ECG-LVSD-Chagas_Sami-Trop_data.csv"
    trace_csv = f"{base}/AI-ECG-LVSD-Chagas_Sami-Trop_data_trace.csv"

    lab = pd.read_csv(labels_csv, sep=";", dtype=str)
    for c in lab.columns:
        if c != "ID":
            lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["ID_exam"] = lab["ID_exam"].astype(np.int64).astype(str)
    print(f"anotaciones: {len(lab)} pacientes seropositivos confirmados")

    print("cargando y preprocesando trazados...")
    x, ids = cargar_trazados(trace_csv, set(lab["ID_exam"]))
    print(f"  {len(x)} registros -> {x.shape}")

    lab = lab.set_index("ID_exam").loc[ids].reset_index()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    n_pat = ckpt["modelo"]["cabeza_patrones.weight"].shape[0] if "cabeza_patrones.weight" in ckpt["modelo"] else 0
    modelo = ResNet1D(n_patrones=n_pat).to(device)
    modelo.load_state_dict(ckpt["modelo"])
    print(f"checkpoint: {args.checkpoint} (epoca {ckpt.get('epoca')}, {n_pat} cabezas de patron)")

    pred = predecir(modelo, x, device, args.batch)

    print(f"\n{'cabeza':<8} {'anotacion':<32} {'n pos':>6} {'prev':>7} {'AUC':>8} {'AUPRC':>8}")
    print("-" * 76)
    filas = []
    for cabeza, (col, nombre) in VALIDACIONES.items():
        if cabeza not in pred:
            continue
        y = lab[col].to_numpy(dtype=np.float32)
        s = pred[cabeza]
        ok = np.isfinite(y)
        y, s = y[ok], s[ok]
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        print(f"{cabeza:<8} {nombre:<32} {int(y.sum()):>6} {y.mean()*100:>6.1f}% {auc:>8.4f} {ap:>8.4f}")
        filas.append((cabeza, auc, ap))

    # Control de cordura: la cabeza de Chagas sobre una cohorte 100% positiva no tiene AUC
    # (falta una clase), pero su score medio dice si el modelo los reconoce como sospechosos.
    print(f"\ncabeza de Chagas sobre los {len(pred['chagas'])} (todos positivos): "
          f"score medio {pred['chagas'].mean():.4f}, mediana {np.median(pred['chagas']):.4f}")

    # LVEF: el label de cardiopatia que ninguna cabeza fue entrenada para predecir. Se mide
    # como exploracion, NO como validacion de nada.
    lvef = lab["V2"].to_numpy(dtype=np.float32)
    lvsd = (lvef <= 40).astype(np.float32)
    if lvsd.sum() >= 5:
        print(f"\nEXPLORATORIO -- disfuncion sistolica (LVEF<=40, {int(lvsd.sum())} casos, "
              f"{lvsd.mean()*100:.1f}%), ninguna cabeza fue entrenada para esto:")
        for nombre, s in pred.items():
            print(f"   {nombre:<8} AUC {roc_auc_score(lvsd, s):.4f}")


if __name__ == "__main__":
    main()
