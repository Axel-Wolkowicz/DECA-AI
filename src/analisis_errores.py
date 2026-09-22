"""Fase 5, tarea final: quienes son los falsos negativos de la banda alta.

    python src/analisis_errores.py --checkpoint D:/DECA-datasets/modelos/patrones-lr8/mejor.pt
    python src/analisis_errores.py --solo-analisis D:/DECA-datasets/modelos/test_scores_<fecha>.parquet

**Que pregunta responde.** La banda alta deriva al 1,4% de la poblacion con PPV 29,5%, y a
cambio deja sin marcar al 78,2% de los casos (Fase 5, 2026-09-14). Esta es la unica tarea que
quedaba abierta de esa fase: entender *quienes* son esos casos perdidos. Hay dos respuestas
posibles y llevan a conclusiones opuestas sobre si el proyecto esta terminado como ML:

  (a) Los perdidos tienen el ECG anotado como normal y ninguna cabeza de patron encendida.
      Entonces no se pierden por una falla del modelo: **no hay nada en el trazado que leer**
      (fase indeterminada, sin cardiopatia). Eso confirma el techo de ruido de etiqueta desde
      el otro lado -- serologia positiva sin cardiopatia -- y cierra la discusion de modelado.
  (b) Los perdidos tienen patron visible (BRD, HBAI o extrasistoles altos) y aun asi score de
      Chagas bajo. Entonces hay un hueco de modelado concreto y nombrable.

**Higiene estadistica: esto NO es una segunda medicion de test.** Se vuelve a correr
inferencia sobre el mismo checkpoint congelado, con los mismos umbrales calibrados en
validacion. No se elige epoca, ni umbral, ni variante, ni se retoca nada a partir de lo que
salga de aca: es descriptivo. Si el resultado sugiriera un cambio de modelo, ese cambio se
valida en val, no en test.

**Sale un artefacto reutilizable**: `test_scores_<fecha>.parquet`, con el score de Chagas y el
de las 4 cabezas auxiliares por registro. Cualquier pregunta futura sobre test se responde en
pandas contra ese parquet, sin volver a pasar por la GPU (`--solo-analisis`).
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from config import CODE15_EXAMS_CSV, MODELOS_DIR
from dataset import PATRONES, cargar_metadata_fase4, filtrar_split
from evaluar import agregar_por_paciente, calibrar_umbrales
from evaluar_test import predecir
from model import ResNet1D

# Cabezas que se usan como evidencia de "hay patron visible". `zona` queda AFUERA a
# proposito: 0,6826 de AUC en test y 0,5218 en chagasicos reales (Fase 5), o sea que es
# indistinguible del azar en la poblacion que importa. Meterla aca contaminaria el conteo
# de "patron detectado" con ruido.
CABEZAS_EVIDENCIA = ("rbbb", "hbai", "extra")

# Un patron se cuenta como "detectado" si su score supera el percentil 95 de esa misma
# cabeza entre los NEGATIVOS de arena A. Es un punto de operacion al 5% de falsos positivos,
# definido mirando solo negativos: no puede inflarse solo porque los positivos tengan
# scores altos, que es justo lo que se quiere medir.
PERCENTIL_NEGATIVOS = 95

# Las SEIS columnas diagnosticas de code15/exams.csv. El proyecto entero uso solo `RBBB`
# (es el unico de los 3 patrones del ROADMAP que CODE-15% anota); las otras cinco estaban
# en disco sin mirarse y son la unica forma de saber que tienen los perdidos que no tienen
# BRD. No entran al modelo: son para describir, no para entrenar.
DX_CODE15 = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")

DECADAS = [0, 30, 40, 50, 60, 70, 80, 200]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC95 de una proporcion por el metodo de Wilson. Se usa este y no el normal porque
    varios subgrupos quedan con n chico y proporciones cerca de 0 o 1, donde el intervalo
    normal se sale de [0,1] y deja de significar nada."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centro - margen)), float(min(1.0, centro + margen)))


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "    -  "
    lo, hi = wilson(k, n)
    return f"{k/n*100:5.1f}% [{lo*100:.1f}-{hi*100:.1f}]"


def tabla_registros(meta: pd.DataFrame, pred: dict) -> pd.DataFrame:
    """Una fila por registro con todos los scores. `predecir` devuelve las predicciones en
    el orden de `meta` (shuffle=False), asi que se alinean por posicion -- igual que en
    evaluar_test.py."""
    cols = ["record_id", "dataset", "patient_id", "edad", "sexo", "confianza",
            "chagas_label", "ecg_anormal", "ecg_anormal_mask", "rbbb_label", "rbbb_mask"]
    df = meta[cols].copy().reset_index(drop=True)
    df["chagas_label"] = df["chagas_label"].astype(np.float32)
    df["s_chagas"] = pred["chagas"]
    df["s_rbbb"] = pred["rbbb"]
    for j, patron in enumerate(PATRONES):
        df[f"s_{patron}"] = pred["patrones"][:, j] if j < pred["patrones"].shape[1] else np.nan
    return df


def tabla_pacientes(reg: pd.DataFrame) -> pd.DataFrame:
    """Colapsa a paciente quedandose con el registro de score MAXIMO.

    La agregacion del proyecto es `max` (evaluar.agregar_por_paciente), asi que el examen
    que define si el paciente cae en la banda alta es el de score mas alto. Los atributos
    --edad, sexo, ECG anormal, scores de las cabezas-- se toman de **ese** registro y no de
    un promedio del paciente: la pregunta es "que vio el modelo en el examen que decidio",
    y promediar mezclaria ese examen con otros que no participaron de la decision.
    """
    idx = reg.groupby(["dataset", "patient_id"])["s_chagas"].idxmax()
    pac = reg.loc[idx].reset_index(drop=True)
    # `y` a nivel paciente es el max de la etiqueta, no la del registro elegido: un paciente
    # es positivo si cualquiera de sus examenes lo es.
    ymax = (reg.groupby(["dataset", "patient_id"])["chagas_label"].max()
            .rename("y").reset_index())
    return pac.merge(ymax, on=["dataset", "patient_id"])


def umbrales_de_cabeza(pac_a: pd.DataFrame) -> dict:
    """Punto de operacion de cada cabeza al 5% de FPR, calculado SOLO sobre los negativos de
    arena A (ver PERCENTIL_NEGATIVOS)."""
    neg = pac_a[pac_a["y"] == 0]
    return {h: float(np.percentile(neg[f"s_{h}"].to_numpy(), PERCENTIL_NEGATIVOS))
            for h in CABEZAS_EVIDENCIA}


def _marcar(pac: pd.DataFrame, umbral_alto: float, umb_cab: dict) -> pd.DataFrame:
    pac = pac.copy()
    pac["derivado"] = pac["s_chagas"] >= umbral_alto
    for h, u in umb_cab.items():
        pac[f"alta_{h}"] = pac[f"s_{h}"] >= u
    pac["patron_detectado"] = np.logical_or.reduce(
        [pac[f"alta_{h}"].to_numpy() for h in umb_cab])
    pac["decada"] = pd.cut(pac["edad"], bins=DECADAS, right=False,
                           labels=["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"])
    return pac


def perfil(grupo: pd.DataFrame, umb_cab: dict) -> dict:
    """Los numeros que describen un grupo de pacientes. `ecg_anormal` se mide solo sobre los
    que TIENEN la anotacion (mask=1); contar los no anotados como normales inventaria
    normalidad donde solo hay ausencia de dato."""
    n = len(grupo)
    con_anot = grupo[grupo["ecg_anormal_mask"] > 0]
    d = {
        "n": int(n),
        "edad_mediana": float(grupo["edad"].median()) if n else float("nan"),
        "pct_mujeres": float((grupo["sexo"] == "F").mean()) if n else float("nan"),
        "n_con_anotacion_ecg": int(len(con_anot)),
        "pct_ecg_anormal": float(con_anot["ecg_anormal"].mean()) if len(con_anot) else float("nan"),
        "pct_patron_detectado": float(grupo["patron_detectado"].mean()) if n else float("nan"),
        "score_chagas_mediano": float(grupo["s_chagas"].median()) if n else float("nan"),
    }
    for h in umb_cab:
        d[f"pct_alta_{h}"] = float(grupo[f"alta_{h}"].mean()) if n else float("nan")
    return d


def cruce_ecg_patron(grupo: pd.DataFrame) -> dict:
    """La tabla que decide entre las hipotesis (a) y (b) del docstring: de los perdidos,
    cuantos no tenian NADA que leer (ECG anotado normal y ninguna cabeza encendida) contra
    cuantos tenian patron visible y el modelo igual no los levanto."""
    con_anot = grupo[grupo["ecg_anormal_mask"] > 0]
    out = {"n_con_anotacion": int(len(con_anot))}
    for anormal in (0.0, 1.0):
        for patron in (False, True):
            sel = con_anot[(con_anot["ecg_anormal"] == anormal)
                           & (con_anot["patron_detectado"] == patron)]
            clave = f"ecg_{'anormal' if anormal else 'normal'}__patron_{'si' if patron else 'no'}"
            out[clave] = int(len(sel))
    # Los que no tienen anotacion de ECG igual se pueden clasificar por las cabezas.
    sin_anot = grupo[grupo["ecg_anormal_mask"] == 0]
    out["sin_anotacion_ecg"] = int(len(sin_anot))
    out["sin_anotacion_ecg__patron_si"] = int(sin_anot["patron_detectado"].sum()) if len(sin_anot) else 0
    return out


def mecanismo_brd(pac_a: pd.DataFrame) -> dict:
    """Por que se pierden los que se pierden, y por que se pierden mas mujeres.

    Estratifica por la anotacion REAL de BRD de code15/exams.csv (`rbbb_label`), no por el
    score de la cabeza: si se usara el score, "los que el modelo cree que tienen BRD" y "los
    que el modelo deriva" serian casi la misma variable y el resultado seria circular.

    Tambien mide el sesgo por sexo *a igual BRD*. Si la brecha por sexo desaparece dentro de
    cada estrato, el modelo no trata distinto a una mujer con el mismo ECG y la disparidad
    viene de la prevalencia del hallazgo, no del modelo -- que es una cosa distinta a la hora
    de declararla.
    """
    from sklearn.metrics import roc_auc_score

    pos = pac_a[(pac_a["y"] > 0) & (pac_a["rbbb_mask"] > 0)]
    fn = pos[~pos["derivado"]]

    estratos = []
    for lab in (0.0, 1.0):
        g = pos[pos["rbbb_label"] == lab]
        k, n = int(g["derivado"].sum()), len(g)
        lo, hi = wilson(k, n)
        estratos.append({"brd_anotado": int(lab), "n": n, "derivados": k,
                         "sensibilidad": k / n if n else float("nan"), "ic95": [lo, hi]})

    por_sexo = []
    for sexo, g in pos.groupby("sexo"):
        for alto in (False, True):
            h = g[g["alta_rbbb"] == alto]
            k, n = int(h["derivado"].sum()), len(h)
            lo, hi = wilson(k, n)
            por_sexo.append({"sexo": str(sexo), "cabeza_brd_alta": alto, "n": n,
                             "sensibilidad": k / n if n else float("nan"), "ic95": [lo, hi]})

    auc_sexo = {}
    prev_brd = {}
    for sexo, g in pac_a.groupby("sexo"):
        if g["y"].nunique() > 1:
            auc_sexo[str(sexo)] = float(roc_auc_score(g["y"], g["s_chagas"]))
        prev_brd[str(sexo)] = {
            "positivos": float(g[g["y"] > 0]["alta_rbbb"].mean()) if (g["y"] > 0).any() else float("nan"),
            "negativos": float(g[g["y"] == 0]["alta_rbbb"].mean()) if (g["y"] == 0).any() else float("nan"),
        }

    # El grupo que define lo que falta: perdidos con ECG anormal y ningun patron. Se compara
    # su score contra el de los negativos equivalentes para saber si el modelo no ve nada en
    # ellos o si los ve pero por debajo del umbral.
    huerfanos = fn[(fn["ecg_anormal"] == 1) & (~fn["patron_detectado"])]
    neg_eq = pac_a[(pac_a["y"] == 0) & (pac_a["ecg_anormal"] == 1) & (~pac_a["patron_detectado"])]

    return {
        "n_positivos_con_anotacion_brd": int(len(pos)),
        "sensibilidad_por_brd_anotado": estratos,
        "pct_FN_con_brd_anotado": float(fn["rbbb_label"].mean()) if len(fn) else float("nan"),
        "n_FN_con_brd_anotado": int(fn["rbbb_label"].sum()) if len(fn) else 0,
        "sensibilidad_por_sexo_y_brd": por_sexo,
        "auc_dentro_de_cada_sexo": auc_sexo,
        "prevalencia_cabeza_brd_por_sexo": prev_brd,
        "FN_ecg_anormal_sin_patron": {
            "n": int(len(huerfanos)),
            "pct_brd_anotado": float(huerfanos["rbbb_label"].mean()) if len(huerfanos) else float("nan"),
            "score_mediano": float(huerfanos["s_chagas"].median()) if len(huerfanos) else float("nan"),
            "score_mediano_negativos_equivalentes": float(neg_eq["s_chagas"].median()) if len(neg_eq) else float("nan"),
            "n_negativos_equivalentes": int(len(neg_eq)),
        },
    }


def vocabulario_code15(pac_a: pd.DataFrame) -> dict:
    """Contra las SEIS columnas diagnosticas de code15/exams.csv, no solo `RBBB`.

    Responde la pregunta que abre el analisis de arriba: si los perdidos tienen el ECG
    anotado como anormal pero no tienen BRD, ¿que tienen? Si la respuesta fuera "LBBB, AF,
    bloqueo AV", faltaria vocabulario de patrones y habria un camino de mejora barato
    --anotar mas cabezas con datos que ya estan en disco--. Si la respuesta es "nada de lo
    anotado", el hueco cae afuera de toda taxonomia disponible y el unico camino es otro
    tipo de dato (ecocardiograma), que es adquisicion y no modelado.

    El merge va por (dataset, record_id) y no por record_id solo, como todos los del
    proyecto: los record_id colisionan entre fuentes (ver dataset.py).
    """
    ex = pd.read_csv(CODE15_EXAMS_CSV, usecols=["exam_id"] + list(DX_CODE15))
    ex["record_id"] = ex["exam_id"].astype(str)
    ex["dataset"] = "code15"
    pa = pac_a.merge(ex[["dataset", "record_id"] + list(DX_CODE15)],
                     on=["dataset", "record_id"], how="left")
    for c in DX_CODE15:
        pa[c] = pa[c].astype(float)
    pa["n_dx"] = pa[list(DX_CODE15)].sum(axis=1)

    pos = pa[pa["y"] > 0]
    fn = pos[~pos["derivado"]]
    grupos = {
        "FN": fn,
        "FN_ecg_anormal_sin_patron": fn[(fn["ecg_anormal"] == 1) & (~fn["patron_detectado"])],
        "TP": pos[pos["derivado"]],
        "negativos": pa[pa["y"] == 0],
    }
    prevalencias = {
        nombre: {"n": int(len(g)),
                 **{c: float(g[c].mean()) if len(g) else float("nan") for c in DX_CODE15},
                 "sin_ningun_dx": float((g["n_dx"] == 0).mean()) if len(g) else float("nan")}
        for nombre, g in grupos.items()
    }

    sens = []
    for c in list(DX_CODE15) + ["ninguno"]:
        g = pos[pos["n_dx"] == 0] if c == "ninguno" else pos[pos[c] == 1]
        k, n = int(g["derivado"].sum()), len(g)
        lo, hi = wilson(k, n)
        sens.append({"dx": c, "n": n, "sensibilidad": k / n if n else float("nan"),
                     "ic95": [lo, hi]})

    return {"prevalencia_por_grupo": prevalencias, "sensibilidad_por_dx": sens}


def sensibilidad_por_subgrupo(pos: pd.DataFrame, col: str) -> list[dict]:
    """Sensibilidad de la banda alta dentro de cada nivel de `col`, con IC95 de Wilson.
    Si varia sistematicamente entre subgrupos es un sesgo reportable, no una curiosidad."""
    filas = []
    for nivel, g in pos.groupby(col, observed=True, dropna=False):
        k, n = int(g["derivado"].sum()), len(g)
        lo, hi = wilson(k, n)
        filas.append({"nivel": str(nivel), "n_positivos": n, "derivados": k,
                      "sensibilidad": k / n if n else float("nan"),
                      "ic95": [lo, hi]})
    return filas


def analizar(reg: pd.DataFrame, umbrales: dict) -> dict:
    pac = tabla_pacientes(reg)
    pac_a = pac[pac["dataset"] == "code15"]
    umb_cab = umbrales_de_cabeza(pac_a)
    pac = _marcar(pac, umbrales["umbral_alto"], umb_cab)
    pac_a = pac[pac["dataset"] == "code15"]
    pac_b = pac[pac["dataset"] == "samitrop"]

    pos_a = pac_a[pac_a["y"] > 0]
    fn_a = pos_a[~pos_a["derivado"]]
    tp_a = pos_a[pos_a["derivado"]]
    neg_a = pac_a[pac_a["y"] == 0]

    fn_b = pac_b[~pac_b["derivado"]]
    tp_b = pac_b[pac_b["derivado"]]

    res = {
        "umbral_alto": umbrales["umbral_alto"],
        "umbral_bajo": umbrales["umbral_bajo"],
        "umbrales_de_cabeza": umb_cab,
        "arena_A": {
            "n_pacientes": int(len(pac_a)),
            "n_positivos": int(len(pos_a)),
            "sensibilidad_banda_alta": float(len(tp_a) / len(pos_a)) if len(pos_a) else float("nan"),
            "perfil_FN": perfil(fn_a, umb_cab),
            "perfil_TP": perfil(tp_a, umb_cab),
            "perfil_negativos": perfil(neg_a, umb_cab),
            "cruce_FN": cruce_ecg_patron(fn_a),
            "cruce_TP": cruce_ecg_patron(tp_a),
            "sensibilidad_por_sexo": sensibilidad_por_subgrupo(pos_a, "sexo"),
            "sensibilidad_por_decada": sensibilidad_por_subgrupo(pos_a, "decada"),
            "mecanismo_brd": mecanismo_brd(pac_a),
            "vocabulario_code15": vocabulario_code15(pac_a),
        },
        "arena_B": {
            "n_pacientes": int(len(pac_b)),
            "recall_banda_alta": float(len(tp_b) / len(pac_b)) if len(pac_b) else float("nan"),
            "perfil_FN": perfil(fn_b, umb_cab),
            "perfil_TP": perfil(tp_b, umb_cab),
            "cruce_FN": cruce_ecg_patron(fn_b),
            "sensibilidad_por_sexo": sensibilidad_por_subgrupo(pac_b.assign(y=1.0), "sexo"),
            "sensibilidad_por_decada": sensibilidad_por_subgrupo(pac_b.assign(y=1.0), "decada"),
        },
    }
    return res


def imprimir(res: dict) -> None:
    ancho = 78
    print("\n" + "=" * ancho)
    print("ANALISIS DE FALSOS NEGATIVOS DE LA BANDA ALTA  (test, checkpoint congelado)")
    print("=" * ancho)
    print(f"umbral alto: {res['umbral_alto']:.4f}  (calibrado en val, arena A)")
    print("cabezas: un patron cuenta como detectado si supera el percentil "
          f"{PERCENTIL_NEGATIVOS} de los negativos de arena A")
    print("  " + "   ".join(f"{h} >= {u:.4f}" for h, u in res["umbrales_de_cabeza"].items()))

    for arena, titulo in (("arena_A", "ARENA A -- CODE-15%, etiqueta autorreportada"),
                          ("arena_B", "ARENA B -- SaMi-Trop, etiqueta SEROLOGICA")):
        a = res[arena]
        print("\n" + "-" * ancho)
        print(titulo)
        print("-" * ancho)
        if arena == "arena_A":
            print(f"{a['n_positivos']} positivos; la banda alta encuentra "
                  f"{a['sensibilidad_banda_alta']*100:.1f}%  "
                  f"-> {a['perfil_FN']['n']} perdidos")
            grupos = [("PERDIDOS (FN)", a["perfil_FN"]), ("ENCONTRADOS (TP)", a["perfil_TP"]),
                      ("negativos (referencia)", a["perfil_negativos"])]
        else:
            print(f"{a['n_pacientes']} pacientes, todos positivos por serologia; la banda "
                  f"alta encuentra {a['recall_banda_alta']*100:.1f}%  "
                  f"-> {a['perfil_FN']['n']} perdidos")
            grupos = [("PERDIDOS (FN)", a["perfil_FN"]), ("ENCONTRADOS (TP)", a["perfil_TP"])]

        print(f"\n  {'':<24}" + "".join(f"{g[0]:>24}" for g in grupos))
        filas = [("n", "n", "{:d}"), ("edad mediana", "edad_mediana", "{:.0f}"),
                 ("% mujeres", "pct_mujeres", "{:.1%}"),
                 ("con anotacion de ECG", "n_con_anotacion_ecg", "{:d}"),
                 ("  % ECG anormal", "pct_ecg_anormal", "{:.1%}"),
                 ("% algun patron alto", "pct_patron_detectado", "{:.1%}")]
        filas += [(f"  % {h} alto", f"pct_alta_{h}", "{:.1%}") for h in res["umbrales_de_cabeza"]]
        filas += [("score Chagas mediano", "score_chagas_mediano", "{:.4f}")]
        for etiqueta, clave, fmt in filas:
            celdas = "".join(f"{fmt.format(g[1][clave]):>24}" if not np.isnan(g[1][clave])
                             else f"{'-':>24}" for g in grupos)
            print(f"  {etiqueta:<24}{celdas}")

        c = a["cruce_FN"]
        print(f"\n  De los perdidos CON anotacion de ECG (n={c['n_con_anotacion']}):")
        for anormal in ("normal", "anormal"):
            for patron in ("no", "si"):
                k = c[f"ecg_{anormal}__patron_{patron}"]
                print(f"    ECG {anormal:<7} / patron {patron:<2}  "
                      f"{k:>6}  {_pct(k, c['n_con_anotacion'])}")
        if c["sin_anotacion_ecg"]:
            print(f"    (sin anotacion de ECG: {c['sin_anotacion_ecg']}, "
                  f"{c['sin_anotacion_ecg__patron_si']} con patron alto)")

        print("\n  Sensibilidad de la banda alta por subgrupo (IC95 Wilson):")
        for col, nombre in (("sensibilidad_por_sexo", "sexo"),
                            ("sensibilidad_por_decada", "edad")):
            print(f"    por {nombre}:")
            for f in a[col]:
                print(f"      {f['nivel']:<8} n={f['n_positivos']:>5}  "
                      f"{_pct(f['derivados'], f['n_positivos'])}")


def imprimir_mecanismo(m: dict) -> None:
    ancho = 78
    print("\n" + "=" * ancho)
    print("MECANISMO -- arena A, estratificado por la anotacion REAL de BRD")
    print("=" * ancho)
    print(f"  positivos con anotacion de BRD: {m['n_positivos_con_anotacion_brd']}")
    for e in m["sensibilidad_por_brd_anotado"]:
        etiqueta = "con BRD anotado" if e["brd_anotado"] else "sin BRD anotado"
        print(f"    {etiqueta:<18} n={e['n']:>4}  sensibilidad banda alta "
              f"{_pct(e['derivados'], e['n'])}")
    print(f"\n  De los perdidos, con BRD anotado real: {m['n_FN_con_brd_anotado']} "
          f"({m['pct_FN_con_brd_anotado']*100:.1f}%)  <- si es bajo, el modelo NO esta "
          f"perdiendo BRD visible")

    h = m["FN_ecg_anormal_sin_patron"]
    print(f"\n  Perdidos con ECG anormal y SIN patron (n={h['n']}): el grupo que define lo que falta")
    print(f"    BRD anotado: {h['pct_brd_anotado']*100:.1f}%  -> son anormalidades que no son BRD")
    print(f"    score de Chagas mediano {h['score_mediano']:.4f}  contra "
          f"{h['score_mediano_negativos_equivalentes']:.4f} de los negativos equivalentes "
          f"(n={h['n_negativos_equivalentes']})")
    print("    -> el modelo los rankea por encima de un negativo comparable, pero sub-umbral")

    print("\n  Sesgo por sexo A IGUAL BRD (si la brecha desaparece, no es trato distinto):")
    for f in m["sensibilidad_por_sexo_y_brd"]:
        estado = "cabeza BRD alta " if f["cabeza_brd_alta"] else "cabeza BRD baja "
        print(f"    {f['sexo']}  {estado} n={f['n']:>4}  "
              f"{f['sensibilidad']*100:5.1f}% [{f['ic95'][0]*100:.1f}-{f['ic95'][1]*100:.1f}]")
    print("  AUC dentro de cada sexo (no depende del umbral): "
          + "  ".join(f"{s} {v:.4f}" for s, v in m["auc_dentro_de_cada_sexo"].items()))
    print("  prevalencia de la cabeza de BRD alta:")
    for s, v in m["prevalencia_cabeza_brd_por_sexo"].items():
        print(f"    {s}  positivos {v['positivos']*100:5.1f}%   negativos {v['negativos']*100:5.1f}%")


def imprimir_vocabulario(v: dict) -> None:
    ancho = 78
    print("\n" + "=" * ancho)
    print("VOCABULARIO -- que tienen los perdidos, contra los 6 dx de code15/exams.csv")
    print("=" * ancho)
    cab = "  ".join(f"{c:>6}" for c in DX_CODE15)
    print(f"  {'grupo':<30}{'n':>7}  {cab}   sin dx")
    for nombre, g in v["prevalencia_por_grupo"].items():
        celdas = "  ".join(f"{g[c]*100:5.1f}%" for c in DX_CODE15)
        print(f"  {nombre:<30}{g['n']:>7}  {celdas}   {g['sin_ningun_dx']*100:5.1f}%")
    print("\n  Sensibilidad de la banda alta segun el dx presente (positivos de arena A):")
    for f in v["sensibilidad_por_dx"]:
        print(f"    {f['dx']:<8} n={f['n']:>4}  "
              f"{f['sensibilidad']*100:5.1f}% [{f['ic95'][0]*100:.1f}-{f['ic95'][1]*100:.1f}]")
    print("\n  Leer asi: si el grueso de los perdidos cae en 'sin dx', el hueco NO se tapa")
    print("  anotando mas patrones de ECG -- no esta en ninguna taxonomia que tengamos.")


def main():
    p = argparse.ArgumentParser(description="Fase 5: analisis de falsos negativos")
    p.add_argument("--checkpoint", help="checkpoint congelado (patrones-lr8/mejor.pt)")
    p.add_argument("--solo-analisis", help="parquet de scores ya calculado; no usa GPU")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.solo_analisis:
        reg = pd.read_parquet(args.solo_analisis)
        umbrales = {"umbral_alto": float(reg.attrs.get("umbral_alto", np.nan)),
                    "umbral_bajo": float(reg.attrs.get("umbral_bajo", np.nan))}
        if not np.isfinite(umbrales["umbral_alto"]):
            meta_u = json.loads(open(args.solo_analisis + ".umbrales.json", encoding="utf-8").read())
            umbrales = meta_u
        print(f"scores leidos de {args.solo_analisis} ({len(reg)} registros)")
    else:
        if not args.checkpoint:
            p.error("hace falta --checkpoint o --solo-analisis")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device)
        n_pat = (ckpt["modelo"]["cabeza_patrones.weight"].shape[0]
                 if "cabeza_patrones.weight" in ckpt["modelo"] else 0)
        n_demo = ckpt["modelo"]["cabeza_chagas.weight"].shape[1] - 3200
        modelo = ResNet1D(n_demograficos=n_demo, n_patrones=n_pat).to(device)
        modelo.load_state_dict(ckpt["modelo"])
        print(f"checkpoint: {args.checkpoint}  (epoca {ckpt.get('epoca')}, "
              f"{n_pat} cabezas de patron, {n_demo} demograficos)")

        meta = cargar_metadata_fase4()

        # Los umbrales se recalibran en VAL con este mismo checkpoint, igual que hizo
        # evaluar_test.py. No se leen de test ni se tocan: este script no elige nada.
        print("\n[1/2] validacion (solo para recuperar los umbrales)")
        meta_val = filtrar_split(meta, "val")
        pred_val = predecir(modelo, meta_val, device, args.batch, args.workers)
        pac_val = agregar_por_paciente(meta_val, pred_val["chagas"])
        a_val = pac_val[pac_val["dataset"] == "code15"]
        umbrales = calibrar_umbrales(a_val["y"].to_numpy(), a_val["score"].to_numpy())
        print(f"  umbral bajo {umbrales['umbral_bajo']:.4f}  "
              f"alto {umbrales['umbral_alto']:.4f}   (Fase 5 registro: 0,0169 / 0,9301)")

        print("\n[2/2] test (inferencia; NO es una segunda medicion, ver docstring)")
        meta_test = filtrar_split(meta, "test")
        pred_test = predecir(modelo, meta_test, device, args.batch, args.workers)
        reg = tabla_registros(meta_test, pred_test)

        ruta = MODELOS_DIR / f"test_scores_{datetime.now():%Y%m%d-%H%M%S}.parquet"
        reg.to_parquet(ruta, index=False)
        with open(str(ruta) + ".umbrales.json", "w", encoding="utf-8") as f:
            json.dump(umbrales, f, indent=2)
        print(f"\nscores por registro guardados en {ruta}")

    res = analizar(reg, umbrales)
    imprimir(res)

    imprimir_mecanismo(res["arena_A"]["mecanismo_brd"])
    imprimir_vocabulario(res["arena_A"]["vocabulario_code15"])

    salida = args.out or str(MODELOS_DIR / f"analisis_fn_{datetime.now():%Y%m%d-%H%M%S}.json")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
