"""Fase 4: ensemble por promedio de scores sobre varios checkpoints ya entrenados.

**No reentrena nada.** Es el punto 1 de "Que queda pendiente" del 2026-08-26, nunca
ejecutado: el paper de referencia (PLOS NTD 2023) promedia 15 modelos y el 5o puesto del
Moody Challenge 2025 promedia 5; todas nuestras corridas fueron de 1 modelo solo.

Por que es lo mas barato que queda: la sesion 18 midio que las cuatro corridas del 2x2 son
indistinguibles entre si en la media de las ultimas 10 epocas (0,1676 / 0,1651 / 0,1646 /
0,1669) mientras sus maximos rebotan entre 0,1735 y 0,1828. Eso es la firma de una serie
con varianza alta y señal plana, que es exactamente el caso donde promediar ayuda. Ojo con
la expectativa: promediar baja la varianza de semilla, **no cruza el techo de ruido de
etiqueta** medido el 2026-08-27.

Tres decisiones de implementacion que no son obvias:

1. **Una sola pasada sobre los datos, N forwards por lote.** Leer el HDF5 de Fase 2 (53 GB
   en un SSD por USB) es lo caro, no la GPU: una pasada por modelo multiplicaria el tiempo
   por N sin necesidad. Los modelos van todos a la GPU a la vez (~25 MB cada uno).

2. **La arquitectura de cada checkpoint se deduce de su state_dict, no de su args.json.**
   Los checkpoints no son todos iguales -- `demo-v1` lleva n_demograficos=2 y
   `patrones-lr8` n_patrones=3 -- y leyendo la forma de los pesos no hay manera de
   construir un modelo que no matchee. De paso entran los checkpoints viejos, anteriores a
   que args.json existiera.

3. **Se reportan tres formas de promediar** porque las metricas son todas de ranking
   (AUC, AUPRC, TPR a capacidad) y el ranking del promedio depende de en que escala se
   promedie: `prob` (sigmoid, acotada, es lo que hace el challenge), `logit` (sin acotar,
   le da mas peso a los modelos confiados en las colas) y `rank` (promedio de rangos,
   inmune a que dos modelos esten calibrados distinto). No hay una correcta a priori.

Control incorporado: `abl-peso1` solo tiene que dar AUC 0,8378 / AUPRC 0,17550 en arena A.
Si no reproduce eso, el problema es el pipeline y no el ensemble.

    python src/ensemble.py
    python src/ensemble.py --corridas abl-peso1 patrones-lr8 --salida ens.json
"""
import argparse
import json
from itertools import combinations

import numpy as np
import torch
from scipy.special import expit  # sigmoid estable: 1/(1+exp(-x)) desborda con x muy negativo
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODELOS_DIR
from dataset import (
    ECGDataset,
    cargar_metadata_fase4,
    filtrar_split,
    mascara_arena_cardiopatia,
)
from evaluar import evaluar_arenas
from model import ResNet1D

# Corridas reales (>=8 epocas sobre el corpus completo). Se excluyen a proposito las de
# humo y las de 1 epoca: un modelo malo no aporta diversidad, aporta ruido.
CORRIDAS_DEFAULT = (
    "abl-peso1",
    "abl-peso1-seed123",
    "patrones-lr8",
    "real8ep",
    "demo-v1",
    "abl-peso5",
)
ESCALAS = ("prob", "logit", "rank")


def construir_desde_state_dict(sd: dict) -> ResNet1D:
    """Instancia una ResNet1D con la forma que pidan los pesos. Ver punto 2 del docstring."""
    n_entrada = sd["cabeza_chagas.weight"].shape[1]
    n_patrones = sd["cabeza_patrones.weight"].shape[0] if "cabeza_patrones.weight" in sd else 0

    modelo = ResNet1D(n_patrones=n_patrones)
    n_demograficos = n_entrada - modelo.n_features
    if n_demograficos:
        modelo = ResNet1D(n_demograficos=n_demograficos, n_patrones=n_patrones)
    return modelo


def cargar_modelos(nombres, checkpoint: str, device) -> dict:
    modelos = {}
    for nombre in nombres:
        ruta = MODELOS_DIR / nombre / checkpoint
        if not ruta.exists():
            raise FileNotFoundError(f"no existe {ruta}")
        ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
        sd = ckpt.get("modelo", ckpt)
        modelo = construir_desde_state_dict(sd)
        modelo.load_state_dict(sd)
        modelos[nombre] = modelo.to(device).eval()
        print(f"  {nombre:<20} epoca {str(ckpt.get('epoca', '?')):>2}  "
              f"demo={modelo.n_demograficos}  patrones={modelo.n_patrones}")
    return modelos


@torch.no_grad()
def inferir(modelos: dict, loader, device, amp: bool) -> dict:
    """{nombre: logits (N,)} en el orden del loader. Una pasada de datos, N forwards.

    Se reportan los logits no finitos por modelo antes de devolver nada. Un solo NaN
    rompe roc_auc_score con un error que no dice de que modelo vino, y con N modelos en la
    misma pasada eso son N sospechosos: mas barato decirlo aca que diagnosticarlo despues.
    """
    acumulado = {nombre: [] for nombre in modelos}
    for x, _, _, _, _, demo, _, _, _ in tqdm(loader, desc="inferencia", leave=False):
        x = x.to(device, non_blocking=True)
        demo = demo.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            for nombre, modelo in modelos.items():
                logit, _, _ = modelo(x, demo)
                acumulado[nombre].append(logit.float().cpu().numpy())

    logits = {nombre: np.concatenate(trozos) for nombre, trozos in acumulado.items()}
    print("\nRango de logits por modelo:")
    for nombre, v in logits.items():
        fin = np.isfinite(v)
        aviso = f"   <-- {int((~fin).sum())} NO FINITOS" if not fin.all() else ""
        print(f"  {nombre:<20} [{v[fin].min():>8.2f}, {v[fin].max():>7.2f}]{aviso}")
    return logits


def combinar(logits: dict, escala: str) -> np.ndarray:
    """Promedia los scores de varios modelos. Ver punto 3 del docstring."""
    matriz = np.stack(list(logits.values()))  # (n_modelos, N)
    if escala == "logit":
        return matriz.mean(axis=0)
    if escala == "prob":
        return expit(matriz).mean(axis=0)
    if escala == "rank":
        return np.stack([rankdata(fila) for fila in matriz]).mean(axis=0)
    raise ValueError(f"escala desconocida: {escala} (opciones: {', '.join(ESCALAS)})")


def fila_metricas(meta, scores, mascara=None) -> dict:
    """AUC / AUPRC / TPR@5% de arena A, opcionalmente sobre un subconjunto de filas.

    `mascara` recorta meta y scores con el MISMO indice: las predicciones vienen alineadas
    con meta por posicion (ver ECGDataset) y desalinearlas daria metricas plausibles pero
    sin sentido.
    """
    if mascara is not None:
        meta, scores = meta[mascara].reset_index(drop=True), scores[mascara]
    res = evaluar_arenas(meta, scores)
    a = res.get("arena_A", {})
    if "error" in a:
        return {"auc": float("nan"), "auprc": float("nan"), "tpr5": float("nan"),
                "atajo": float("nan"), "res": res}
    return {
        "auc": a["auc"],
        "auprc": a["auprc"],
        "tpr5": a["capacidad"]["tpr@5%"],
        "atajo": res["diagnostico_atajo"]["delta_vs_arena_A"],
        "res": res,
    }


def main():
    p = argparse.ArgumentParser(description="Fase 4: ensemble de checkpoints ya entrenados")
    p.add_argument("--corridas", nargs="+", default=list(CORRIDAS_DEFAULT))
    p.add_argument("--checkpoint", default="mejor.pt", help="mejor.pt o ultimo.pt")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="test SOLO con el modelo congelado (Fase 3, regla operativa)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="muestra aleatoria, para probar rapido")
    # AMP APAGADO por default, al reves que train.py, y no es un descuido: `demo-v1`
    # produce 24 logits no finitos de 64.247 en fp16 (su rango es [-162, 5,8], mucho mas
    # ancho que el resto de los checkpoints, y desborda). Entrenando eso lo atrapa el
    # GradScaler; evaluando no lo atrapa nadie, y un score corrupto no da error sino un
    # numero plausible. En fp32 la diferencia de logit contra AMP es 0,0037 y la pasada
    # completa tarda ~4 min en vez de ~2: barato para no tener que confiar.
    p.add_argument("--amp", action="store_true", default=False,
                   help="inferencia en fp16 (mas rapido, pero ver el comentario del codigo)")
    p.add_argument("--salida", default=None, help="ruta de un json con los resultados completos")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    meta = cargar_metadata_fase4()
    meta_split = filtrar_split(meta, args.split, limite=args.limit)
    print(f"\n{args.split}: {len(meta_split)} registros "
          f"({', '.join(f'{d}={n}' for d, n in meta_split['dataset'].value_counts().items())})")

    print("\nCheckpoints:")
    modelos = cargar_modelos(args.corridas, args.checkpoint, device)

    loader = DataLoader(
        ECGDataset(meta_split), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    logits = inferir(modelos, loader, device, args.amp)

    mask_card = mascara_arena_cardiopatia(meta_split)
    print(f"\nArena A-cardiopatia: se sacan {int((~mask_card).sum())} registros "
          f"(positivos con ECG normal); los negativos quedan todos.\n")

    filas, resultados = [], {}
    for nombre in args.corridas:
        s = expit(logits[nombre])
        filas.append((nombre, fila_metricas(meta_split, s),
                      fila_metricas(meta_split, s, mask_card)))
    for escala in ESCALAS:
        s = combinar(logits, escala)
        filas.append((f"ENSEMBLE-{escala}", fila_metricas(meta_split, s),
                      fila_metricas(meta_split, s, mask_card)))

    print(f"{'modelo':<22} | {'A-serologia':^24} | {'A-cardiopatia':^16}")
    print(f"{'':<22} | {'AUC':>7} {'AUPRC':>7} {'TPR@5%':>7} | {'AUC':>7} {'AUPRC':>7}")
    print("-" * 76)
    for nombre, ser, car in filas:
        if nombre == f"ENSEMBLE-{ESCALAS[0]}":
            print("-" * 76)
        print(f"{nombre:<22} | {ser['auc']:>7.4f} {ser['auprc']:>7.4f} {ser['tpr5']*100:>6.1f}% "
              f"| {car['auc']:>7.4f} {car['auprc']:>7.4f}")
        resultados[nombre] = {
            "serologia": {k: v for k, v in ser.items() if k != "res"},
            "cardiopatia": {k: v for k, v in car.items() if k != "res"},
            "arenas_serologia": ser["res"],
        }

    # La ganancia de un ensemble viene de que los modelos se equivoquen en lugares
    # distintos. Correlaciones muy altas (>0,95) predicen que no va a haber ganancia.
    print("\nCorrelacion de Spearman entre modelos (baja = mas diversidad = mas ganancia):")
    pares = []
    for a, b in combinations(args.corridas, 2):
        pares.append((float(spearmanr(logits[a], logits[b]).statistic), a, b))
    for rho, a, b in sorted(pares):
        print(f"  {rho:.4f}  {a} <-> {b}")
    print(f"  media {np.mean([r for r, _, _ in pares]):.4f}")

    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            json.dump({"corridas": args.corridas, "split": args.split,
                       "checkpoint": args.checkpoint, "resultados": resultados,
                       "spearman": [{"a": a, "b": b, "rho": r} for r, a, b in pares]},
                      f, indent=2)
        print(f"\nResultados completos en {args.salida}")


if __name__ == "__main__":
    main()
