"""Controles de plausibilidad: ¿lo que llego es un electrocardiograma con sus derivaciones
bien nombradas?

Existe por un hallazgo del 2026-09-22 (FASES.md): al servicio se le dio ruido blanco y
devolvio score 0,615, percentil 92, con total aplomo. La red nunca se entreno para decir
"esto no es un ECG", asi que no lo dice; los controles de `inferencia.py` atajaban señal
corta, corrupta o plana, pero nada de eso distingue un ECG de ruido.

Son dos controles deterministas sobre la señal, sin tocar el modelo. **Un ECG que los pasa
se puntua exactamente igual que antes**, asi que la calibracion y los numeros de test
siguen aplicando tal cual. Umbrales fijados y medidos sobre validacion (nunca test) con
`validar_plausibilidad.py`; el porque de cada uno esta en FASES.md, sesion del 2026-09-28.

1. **Concentracion temporal de la energia QRS** (`no_parece_ecg`). En un ECG la energia de
   5-30 Hz esta amontonada en los complejos QRS (y en las espigas de marcapasos); en el
   ruido esta repartida pareja en el tiempo. Se mide que fraccion de esa energia cae en el
   15% de instantes mas energeticos. Se eligio esto y no un criterio espectral ("cuanta
   energia hay por encima de 40 Hz") porque el espectral **rechazaba ECG de marcapasos**:
   las espigas de estimulacion tienen tanta alta frecuencia como el ruido blanco, y en
   Chagas el marcapasos es frecuente (4% en SaMi-Trop).

2. **Coherencia de las derivaciones de miembros** (`derivaciones_permutadas`). De las 6 de
   miembros solo 2 son independientes: III = II - I, aVR = -(I+II)/2, aVL = I - II/2,
   aVF = II - I/2. Si esas identidades se cumplen casi exacto *pero con los signos que no
   corresponden*, las columnas estan bien adquiridas y mal nombradas -- aVR y aVL cambiadas,
   por ejemplo, que es la trampa de SaMi-Trop LVSD y el error que `lectura_ecg.py` no puede
   ver porque confia en los nombres. Se mide con regresion libre sobre (I, II), asi que no
   depende de unidades ni del factor 1,5 de VR/VL/VF.

   **Lo que NO se rechaza, a proposito:** que las identidades no cierren en absoluto. Pasa en
   ~2% de CODE-15% -- derivaciones adquiridas por grupos en instantes distintos -- y el modelo
   se entreno con esos registros adentro. Tampoco se detecta la inversion de electrodos de
   brazos: el equipo calcula las derivaciones desde los electrodos ya invertidos, asi que las
   identidades se siguen cumpliendo. Y nada de esto puede ver una permutacion entre
   precordiales, que no tienen identidades entre si.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ventana import OUT_FREQ, recortar_padding

# --- control 1 -----------------------------------------------------------------------
# Medido sobre validacion (2026-09-28): los ECG reales dan mediana 0,91 y minimo 0,34 en
# 10.475 registros de las 4 fuentes; ruido blanco, rosa y random walk dan <= 0,33 (con 12
# derivaciones independientes o con 8 y las de miembros reconstruidas). El umbral se pone
# debajo del minimo real: rechazar un ECG verdadero es peor que dejar pasar ruido raro.
UMBRAL_CONCENTRACION = 0.30
_FRACCION_PICO = 0.15
_SUAVIZADO_S = 0.04
_SOS_QRS = butter(3, [5, 30], btype="band", fs=OUT_FREQ, output="sos")

# --- control 2 -----------------------------------------------------------------------
MIEMBROS = ("I", "II", "III", "aVR", "aVL", "aVF")
# Coeficientes de cada derivacion sobre (I, II). Solo importa la direccion: la escala
# absorbe unidades, ganancias y el factor 1,5 de las unipolares sin aumentar.
_IDENTIDADES = {
    "III": (-1.0, 1.0),
    "aVR": (-1.0, -1.0),
    "aVL": (1.0, -0.5),
    "aVF": (-0.5, 1.0),
}
# "Las identidades cierran" (R2 minimo sobre las 4) y "con la direccion equivocada"
# (coseno minimo entre coeficientes ajustados y esperados). En validacion un ECG bien
# nombrado da coseno 1,000; las permutaciones de miembros dan <= 0,32.
R2_COHERENTE = 0.95
COSENO_MINIMO = 0.90


def concentracion_qrs(ventana: np.ndarray) -> float:
    """Fraccion de la energia 5-30 Hz que cae en el 15% de instantes mas energeticos.

    `ventana` es la salida de `procesar_registro`: (2800, 12) a 400 Hz, z-scoreada. Las
    derivaciones planas se excluyen (no aportan energia y no deben diluir la medida). ~0,2
    para ruido estacionario, ~0,9 para un ECG.
    """
    vivas = ventana[:, ventana.std(axis=0) > 1e-8]
    if vivas.shape[1] == 0:
        return 0.0
    energia = (sosfiltfilt(_SOS_QRS, vivas, axis=0) ** 2).sum(axis=1)
    k = int(_SUAVIZADO_S * OUT_FREQ)
    energia = np.convolve(energia, np.ones(k) / k, mode="valid")
    total = energia.sum()
    if total <= 0:
        return 0.0
    pico = np.sort(energia)[::-1][: int(_FRACCION_PICO * len(energia))]
    return float(pico.sum() / total)


def coherencia_miembros(señal: np.ndarray) -> tuple[float, float]:
    """(R2 minimo, coseno minimo) de las 4 identidades sobre la señal cruda (n, 12) canonica.

    Sobre la señal cruda y entera, no sobre la ventana: el z-score por derivacion de la
    Fase 2 cambia la escala de cada una por separado y las identidades dejarian de ser
    lineales en los mismos coeficientes.
    """
    x = recortar_padding(señal).astype(np.float64)
    x = x - x.mean(axis=0)
    base = x[:, :2]
    r2s, cosenos = [], []
    for nombre, esperado in _IDENTIDADES.items():
        y = x[:, MIEMBROS.index(nombre)]
        coef, *_ = np.linalg.lstsq(base, y, rcond=None)
        var = float((y ** 2).sum())
        r2s.append(1.0 - float(((y - base @ coef) ** 2).sum()) / var if var > 0 else 0.0)
        e = np.asarray(esperado)
        norma = float(np.linalg.norm(coef))
        cosenos.append(float(coef @ e) / (norma * float(np.linalg.norm(e))) if norma > 0 else 0.0)
    return min(r2s), min(cosenos)


def evaluar(señal: np.ndarray, ventana: np.ndarray, derivadas: list[str],
            planas: list[str]) -> dict:
    """Corre los dos controles y devuelve las medidas y los motivos de rechazo.

    `señal` es la cruda canonica (n, 12); `ventana`, la de la Fase 2; `derivadas` y
    `planas`, las listas que ya arma el servicio. El control de miembros se salta si alguna
    de las 6 se reconstruyo (las identidades se cumplen por construccion, no informa nada)
    o esta plana (una columna sin señal hace que la regresion no signifique nada).
    """
    conc = concentracion_qrs(ventana)
    res = {"concentracion_qrs": round(conc, 3), "coherencia_miembros": None, "rechazos": []}
    if conc < UMBRAL_CONCENTRACION:
        res["rechazos"].append("no_parece_ecg")

    if not (set(derivadas) | set(planas)) & set(MIEMBROS):
        r2, coseno = coherencia_miembros(señal)
        res["coherencia_miembros"] = {"r2": round(r2, 3), "coseno": round(coseno, 3)}
        if r2 >= R2_COHERENTE and coseno < COSENO_MINIMO:
            res["rechazos"].append("derivaciones_permutadas")
    return res
