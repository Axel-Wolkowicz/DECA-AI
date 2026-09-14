"""Fase 5: evaluacion sobre el conjunto de TEST. **Se corre una sola vez.**

    python src/evaluar_test.py --checkpoint D:/DECA-datasets/modelos/patrones-lr8/mejor.pt

**Por que esto es distinto de todo lo demas.** Cada numero del proyecto hasta acá -- 0,838 de
AUC, 0,1755 de AUPRC, 41,4% de TPR@5%, y el 0,9797 de la cabeza de BRD -- es de
**validacion**. El test son 63.608 registros congelados que ningun modelo vio nunca, ni para
entrenar ni para elegir checkpoint ni para calibrar un umbral. Mirarlo es irreversible: una
vez que se conoce el resultado, cualquier decision posterior tomada a la luz de ese numero lo
contamina y deja de ser una estimacion honesta del desempeño fuera de muestra.

Por eso este script **no tiene perillas de modelo**. No se elige epoca, no se barre un
umbral, no se prueban variantes. Entra un checkpoint ya elegido por val y sale el resultado.

**La regla que implementa (Fase 3, "Regla operativa"): los umbrales se calibran en
VALIDACION, arena A, y se APLICAN a test.** Calibrarlos en test seria elegir el punto de
operacion mirando la respuesta -- el error clasico que infla cualquier metrica operativa.
Aca se recalculan sobre val con este mismo checkpoint en vez de leer los que quedaron
guardados en el .pt: cuesta una pasada de inferencia y elimina la duda de si el checkpoint
guardo los umbrales de la epoca correcta.

**IC95 por bootstrap estratificado** (2.000 remuestreos, seed 42), igual que el analisis del
2026-08-27. En una medicion de un solo tiro el intervalo importa mas que el puntual: sin el
no se puede decir si una diferencia contra val es real o es el tamaño de muestra.
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODELOS_DIR
from dataset import PATRONES, ECGDataset, cargar_metadata_fase4, filtrar_split
from evaluar import agregar_por_paciente, calibrar_umbrales, evaluar_arenas, formatear
from model import ResNet1D

N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 42


@torch.no_grad()
def predecir(modelo, meta: pd.DataFrame, device, batch: int, workers: int) -> dict:
    """Scores por registro, en el orden de `meta`. shuffle=False no es cosmetico: la
    evaluacion por arenas alinea predicciones con meta por posicion."""
    loader = DataLoader(ECGDataset(meta), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    modelo.eval()
    chagas, rbbb, patrones = [], [], []
    for x, _, _, _, _, demo, _, _, _ in tqdm(loader, desc="  inferencia", leave=False):
        x, demo = x.to(device, non_blocking=True), demo.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lc, lr, lp = modelo(x, demo)
        chagas.append(torch.sigmoid(lc.float()).cpu().numpy())
        rbbb.append(torch.sigmoid(lr.float()).cpu().numpy())
        patrones.append(torch.sigmoid(lp.float()).cpu().numpy())
    return {"chagas": np.concatenate(chagas), "rbbb": np.concatenate(rbbb),
            "patrones": np.concatenate(patrones)}


def bootstrap_arena_a(meta: pd.DataFrame, scores: np.ndarray) -> dict:
    """IC95 de AUC, AUPRC y TPR@5% en arena A, remuestreando PACIENTES (no examenes) y
    estratificando por clase para que cada remuestreo conserve la prevalencia."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from evaluar import tpr_a_capacidad

    pac = agregar_por_paciente(meta, scores)
    a = pac[pac["dataset"] == "code15"]
    y, s = a["y"].to_numpy(), a["score"].to_numpy()
    idx_pos, idx_neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)

    rng = np.random.default_rng(SEED_BOOTSTRAP)
    aucs, aps, tprs = [], [], []
    for _ in range(N_BOOTSTRAP):
        sel = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                              rng.choice(idx_neg, len(idx_neg), replace=True)])
        yb, sb = y[sel], s[sel]
        aucs.append(roc_auc_score(yb, sb))
        aps.append(average_precision_score(yb, sb))
        tprs.append(tpr_a_capacidad(yb, sb)["tpr@5%"])

    def ic(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {"auc_ic95": ic(aucs), "auprc_ic95": ic(aps), "tpr@5%_ic95": ic(tprs)}


def auprc_auc(scores, y, mask):
    from sklearn.metrics import average_precision_score, roc_auc_score

    sel = mask > 0
    if sel.sum() == 0 or len(np.unique(y[sel])) < 2:
        return {"auc": float("nan"), "auprc": float("nan"), "n_pos": 0}
    return {"auc": float(roc_auc_score(y[sel], scores[sel])),
            "auprc": float(average_precision_score(y[sel], scores[sel])),
            "n_pos": int(y[sel].sum()), "prevalencia": float(y[sel].mean())}


def main():
    p = argparse.ArgumentParser(description="Fase 5: evaluacion unica sobre test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="JSON de salida (default: MODELOS_DIR/test_<fecha>.json)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    n_pat = (ckpt["modelo"]["cabeza_patrones.weight"].shape[0]
             if "cabeza_patrones.weight" in ckpt["modelo"] else 0)
    n_demo = ckpt["modelo"]["cabeza_chagas.weight"].shape[1] - 3200
    modelo = ResNet1D(n_demograficos=n_demo, n_patrones=n_pat).to(device)
    modelo.load_state_dict(ckpt["modelo"])
    print(f"checkpoint: {args.checkpoint}")
    print(f"  epoca {ckpt.get('epoca')}, {n_pat} cabezas de patron, {n_demo} demograficos")
    print(f"  AUPRC arena A en val al guardarlo: {ckpt.get('auprc_arena_A')}")

    meta = cargar_metadata_fase4()
    meta_val = filtrar_split(meta, "val")
    meta_test = filtrar_split(meta, "test")
    print(f"\nval  {len(meta_val):>7} registros  ({', '.join(f'{d}={n}' for d, n in meta_val['dataset'].value_counts().items())})")
    print(f"test {len(meta_test):>7} registros  ({', '.join(f'{d}={n}' for d, n in meta_test['dataset'].value_counts().items())})")

    # --- Paso 1: calibrar en VAL -------------------------------------------------------
    print("\n[1/2] validacion (solo para calibrar los umbrales)")
    pred_val = predecir(modelo, meta_val, device, args.batch, args.workers)
    pac_val = agregar_por_paciente(meta_val, pred_val["chagas"])
    a_val = pac_val[pac_val["dataset"] == "code15"]
    umbrales = calibrar_umbrales(a_val["y"].to_numpy(), a_val["score"].to_numpy())
    print(f"  umbrales calibrados en val/arena A: bajo {umbrales['umbral_bajo']:.4f}  "
          f"alto {umbrales['umbral_alto']:.4f}")
    res_val = evaluar_arenas(meta_val, pred_val["chagas"])

    # --- Paso 2: aplicar a TEST --------------------------------------------------------
    print("\n[2/2] TEST -- medicion unica")
    pred_test = predecir(modelo, meta_test, device, args.batch, args.workers)
    res_test = evaluar_arenas(meta_test, pred_test["chagas"], umbrales=umbrales)

    print("\n" + "=" * 72)
    print("RESULTADO SOBRE TEST  (umbrales calibrados en validacion, nunca en test)")
    print("=" * 72)
    print(formatear(res_test))

    print("\nIC95 de arena A por bootstrap de pacientes "
          f"({N_BOOTSTRAP} remuestreos, seed {SEED_BOOTSTRAP}):")
    ic = bootstrap_arena_a(meta_test, pred_test["chagas"])
    a_test = res_test["arena_A"]
    print(f"  AUC    {a_test['auc']:.4f}  [{ic['auc_ic95'][0]:.4f} - {ic['auc_ic95'][1]:.4f}]")
    print(f"  AUPRC  {a_test['auprc']:.4f}  [{ic['auprc_ic95'][0]:.4f} - {ic['auprc_ic95'][1]:.4f}]")
    print(f"  TPR@5% {a_test['capacidad']['tpr@5%']*100:.1f}%  "
          f"[{ic['tpr@5%_ic95'][0]*100:.1f}% - {ic['tpr@5%_ic95'][1]*100:.1f}%]")

    print("\nval -> test (la caida esperable es el costo de no haber elegido nada mirando test):")
    for k in ("auc", "auprc"):
        print(f"  arena A {k:<6} val {res_val['arena_A'][k]:.4f}  ->  test {a_test[k]:.4f}  "
              f"({a_test[k] - res_val['arena_A'][k]:+.4f})")
    cv = res_val["arena_A"]["capacidad"]["tpr@5%"]
    ct = a_test["capacidad"]["tpr@5%"]
    print(f"  arena A TPR@5% val {cv*100:.1f}%  ->  test {ct*100:.1f}%  ({(ct-cv)*100:+.1f} pp)")

    # --- Cabezas auxiliares sobre test --------------------------------------------------
    cabezas = {"rbbb": auprc_auc(pred_test["rbbb"],
                                 meta_test["rbbb_label"].to_numpy(np.float32),
                                 meta_test["rbbb_mask"].to_numpy(np.float32))}
    for j, patron in enumerate(PATRONES):
        if j < pred_test["patrones"].shape[1]:
            cabezas[patron] = auprc_auc(pred_test["patrones"][:, j],
                                        meta_test[f"{patron}_label"].to_numpy(np.float32),
                                        meta_test[f"{patron}_mask"].to_numpy(np.float32))
    print("\nCabezas auxiliares sobre test (a nivel registro, nunca sumadas a la de Chagas):")
    for nombre, m in cabezas.items():
        print(f"  {nombre:<6} AUC {m['auc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"n_pos {m['n_pos']}  prev {m.get('prevalencia', 0)*100:.1f}%")

    salida = args.out or str(MODELOS_DIR / f"test_{datetime.now():%Y%m%d-%H%M%S}.json")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "fecha": datetime.now().isoformat(),
                   "umbrales_de_val": umbrales, "arenas_val": res_val,
                   "arenas_test": res_test, "ic95_test_arena_A": ic,
                   "cabezas_test": cabezas,
                   "n_val": len(meta_val), "n_test": len(meta_test)}, f, indent=2)
    print(f"\nGuardado en {salida}")
    print("\nEsta medicion NO se repite. Cualquier cambio posterior al modelo se evalua en "
          "val, y el test solo se vuelve a tocar si se congela un modelo distinto y se "
          "asume el costo estadistico de haber mirado dos veces.")


if __name__ == "__main__":
    main()
