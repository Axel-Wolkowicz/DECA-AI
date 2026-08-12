"""Fase 4/5: metricas por arena, a nivel paciente.

Nunca se reporta una metrica agregada entre datasets como resultado. La razon esta medida
(FASES.md, Fase 4): SaMi-Trop es 100% positivo y PTB-XL 100% negativo, asi que un AUC
calculado sobre la mezcla premia a un modelo que solo aprendio a reconocer de que fuente
viene la señal. Las arenas separan eso:

  A - CODE-15% sola. DECIDE EL GO/NO-GO. Dos clases y una sola fuente, asi que el atajo
      de fuente es imposible por construccion adentro de la arena. Prevalencia a nivel
      paciente 1,91% (val), que es la que asume toda la tabla de operacion de Fase 3 --
      por eso el PPV y las "serologias por caso" son interpretables aca y no en otro lado.
  B - SaMi-Trop. SOLO RECALL. Es la unica fuente con etiqueta serologica, pero no tiene
      un solo negativo: sin negativos no existe AUC, ni especificidad, ni PPV.
  C - PTB-XL. SOLO ESPECIFICIDAD. Poblacion no endemica, otro equipo, otra decada. Mide
      robustez cruzada de poblacion, no discriminacion. El modelo nunca la vio entrenando
      (ver dataset.py), asi que mide generalizacion real.
  D - Serologica ampliada: SaMi-Trop(+) contra CODE-15%(-). La unica forma de obtener un
      AUC con positivos confirmados por serologia. Es valida solo porque esas dos fuentes
      son dificiles de distinguir entre si (AUC de sonda de fuente 0,598, casi azar); si
      esa sonda subiera, esta arena pasa a ser cota superior, no resultado.

Y el **diagnostico de atajo**: AUC sobre todo junto menos AUC de arena A. No es
performance, es la magnitud del atajo de fuente medida en las unidades del go/no-go.

Los umbrales (bajo = 95% de sensibilidad, alto = PPV >= 30%) se calibran SIEMPRE sobre
validacion y SIEMPRE en la arena A, nunca sobre test (Fase 3, "Regla operativa").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

SENSIBILIDAD_OBJETIVO = 0.95
PPV_BANDA_ALTA = 0.30


def agregar_por_paciente(meta: pd.DataFrame, scores: np.ndarray, modo: str = "max") -> pd.DataFrame:
    """Colapsa examenes a pacientes. 32% de los examenes de CODE-15% son repeticiones de
    un paciente ya visto, asi que evaluar por examen infla las metricas (Fase 3,
    decision 3). `max` es el default por ser el conservador clinicamente -- no perder un
    examen anomalo aislado -- al costo de ~2 puntos de especificidad."""
    df = pd.DataFrame({
        "dataset": meta["dataset"].to_numpy(),
        "patient_id": meta["patient_id"].to_numpy(),
        "y": meta["chagas_label"].to_numpy(dtype=np.float32),
        "score": np.asarray(scores, dtype=np.float64),
    })
    agg = {"max": "max", "mean": "mean"}.get(modo)
    if agg is None:
        raise ValueError(f"modo de agregacion desconocido: {modo} (usar 'max' o 'mean')")
    return (
        df.groupby(["dataset", "patient_id"], as_index=False)
        .agg(y=("y", "max"), score=("score", agg))
    )


def calibrar_umbrales(y: np.ndarray, score: np.ndarray) -> dict:
    """Umbral bajo = el mas alto que todavia da >=95% de sensibilidad. Umbral alto = el
    mas bajo que todavia da PPV >=30% (deriva a la mayor cantidad de gente posible sin
    romper la promesa de la banda alta)."""
    fpr, tpr, thr = roc_curve(y, score)
    alcanzan = np.flatnonzero(tpr >= SENSIBILIDAD_OBJETIVO)
    umbral_bajo = float(thr[alcanzan[0]]) if len(alcanzan) else float(score.min())

    precision, _, thr_pr = precision_recall_curve(y, score)
    # precision_recall_curve devuelve un punto mas que umbrales (el ultimo es precision=1,
    # recall=0 y no tiene umbral): se recorta antes de indexar.
    ok = np.flatnonzero(precision[:-1] >= PPV_BANDA_ALTA)
    umbral_alto = float(thr_pr[ok[0]]) if len(ok) else float("inf")

    return {"umbral_bajo": umbral_bajo, "umbral_alto": umbral_alto}


def _en_umbral(y: np.ndarray, score: np.ndarray, umbral: float) -> dict:
    pred = score >= umbral
    tp = float((pred & (y > 0)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y > 0)).sum())
    tn = float((~pred & (y == 0)).sum())
    return {
        "sensibilidad": tp / (tp + fn) if tp + fn else float("nan"),
        "especificidad": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "derivados": (tp + fp) / len(y) if len(y) else float("nan"),
    }


def _discriminacion(y: np.ndarray, score: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "auprc": float("nan")}
    return {"auc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}


def evaluar_arenas(
    meta: pd.DataFrame,
    scores: np.ndarray,
    umbrales: dict | None = None,
    modo_agregacion: str = "max",
) -> dict:
    """Devuelve un dict de metricas por arena. Si `umbrales` es None se calibran sobre la
    arena A de este mismo conjunto (eso es lo correcto en validacion, y lo INCORRECTO en
    test: ahi hay que pasarle los umbrales calibrados en validacion)."""
    pac = agregar_por_paciente(meta, scores, modo_agregacion)
    code15 = pac[pac["dataset"] == "code15"]
    samitrop = pac[pac["dataset"] == "samitrop"]
    ptbxl = pac[pac["dataset"] == "ptbxl"]

    res = {"n_pacientes": {d: int(len(g)) for d, g in pac.groupby("dataset")}}

    if len(code15) == 0 or code15["y"].nunique() < 2:
        # Pasa en corridas de humo con muestras chicas: sin positivos no hay umbral que
        # calibrar. Se devuelve lo que se pueda en vez de romper el entrenamiento entero.
        res["arena_A"] = {"error": "arena A sin las dos clases; no se calibran umbrales"}
        return res

    ya, sa = code15["y"].to_numpy(), code15["score"].to_numpy()
    if umbrales is None:
        umbrales = calibrar_umbrales(ya, sa)
    res["umbrales"] = umbrales

    res["arena_A"] = {
        **_discriminacion(ya, sa),
        "prevalencia": float(ya.mean()),
        "banda_media": _en_umbral(ya, sa, umbrales["umbral_bajo"]),
        "banda_alta": _en_umbral(ya, sa, umbrales["umbral_alto"]),
    }

    if len(samitrop):
        ys, ss = samitrop["y"].to_numpy(), samitrop["score"].to_numpy()
        res["arena_B"] = {
            "n": int(len(ys)),
            "recall_umbral_bajo": float((ss >= umbrales["umbral_bajo"]).mean()),
            "recall_umbral_alto": float((ss >= umbrales["umbral_alto"]).mean()),
        }

    if len(ptbxl):
        sp = ptbxl["score"].to_numpy()
        res["arena_C"] = {
            "n": int(len(sp)),
            "especificidad_umbral_bajo": float((sp < umbrales["umbral_bajo"]).mean()),
            "especificidad_umbral_alto": float((sp < umbrales["umbral_alto"]).mean()),
        }

    if len(samitrop):
        # Arena D: positivos por serologia (SaMi-Trop) contra negativos de CODE-15%. Los
        # positivos autorreportados de CODE-15% se excluyen a proposito: el punto es que
        # TODOS los positivos de esta arena tengan etiqueta fuerte.
        neg = code15[code15["y"] == 0]
        yd = np.concatenate([np.ones(len(samitrop)), np.zeros(len(neg))])
        sd = np.concatenate([samitrop["score"].to_numpy(), neg["score"].to_numpy()])
        res["arena_D"] = {"n_pos": int(len(samitrop)), "n_neg": int(len(neg)), **_discriminacion(yd, sd)}

    pooled = _discriminacion(pac["y"].to_numpy(), pac["score"].to_numpy())
    res["diagnostico_atajo"] = {
        "auc_pooled": pooled["auc"],
        "delta_vs_arena_A": pooled["auc"] - res["arena_A"]["auc"],
    }
    return res


def formatear(res: dict) -> str:
    """Resumen legible para el log de entrenamiento."""
    if "error" in res.get("arena_A", {}):
        return f"  arena A: {res['arena_A']['error']}"

    a, lineas = res["arena_A"], []
    lineas.append(
        f"  A code15    AUC {a['auc']:.4f}  AUPRC {a['auprc']:.4f}  prev {a['prevalencia']*100:.2f}%"
    )
    bm, ba = a["banda_media"], a["banda_alta"]
    lineas.append(
        f"    banda media (sens 95%): espec {bm['especificidad']*100:5.1f}%  "
        f"PPV {bm['ppv']*100:4.1f}%  deriva {bm['derivados']*100:4.1f}%"
    )
    lineas.append(
        f"    banda alta  (PPV>=30%): sens  {ba['sensibilidad']*100:5.1f}%  "
        f"PPV {ba['ppv']*100:4.1f}%  deriva {ba['derivados']*100:4.1f}%"
    )
    if "arena_B" in res:
        b = res["arena_B"]
        lineas.append(
            f"  B samitrop  recall {b['recall_umbral_bajo']*100:5.1f}% (bajo) / "
            f"{b['recall_umbral_alto']*100:5.1f}% (alto)   n={b['n']}"
        )
    if "arena_C" in res:
        c = res["arena_C"]
        lineas.append(
            f"  C ptbxl     espec  {c['especificidad_umbral_bajo']*100:5.1f}% (bajo) / "
            f"{c['especificidad_umbral_alto']*100:5.1f}% (alto)   n={c['n']}"
        )
    if "arena_D" in res:
        d = res["arena_D"]
        lineas.append(f"  D serologia AUC {d['auc']:.4f}  AUPRC {d['auprc']:.4f}  "
                      f"({d['n_pos']}+ vs {d['n_neg']}-)")
    dg = res["diagnostico_atajo"]
    lineas.append(f"  atajo de fuente: AUC pooled {dg['auc_pooled']:.4f}  "
                  f"delta vs A {dg['delta_vs_arena_A']:+.4f}")
    return "\n".join(lineas)
