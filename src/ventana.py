"""El preprocesamiento de señal de la Fase 2, aislado de las rutas del SSD.

Este modulo existe por una razon concreta: `preprocess.py` importa `config.py`, y
`config.py` resuelve la ruta del SSD **al importarse** -- si el disco no esta enchufado,
revienta con FileNotFoundError antes de ejecutar una sola linea util. Eso esta bien para
las Fases 0-2, que no tienen sentido sin los datasets, y es fatal para el servicio de
inferencia: `servidor.py` corre en una maquina que no tiene ni va a tener los 47 GB de
corpus, pero tiene que aplicar **exactamente** el mismo preprocesamiento con el que se
entreno y se midio el modelo.

Ese "exactamente" es el punto entero del modulo. El AUC 0,8348 de test (2026-09-14) se
midio sobre señal recortada a 7,0 s, resampleada a 400 Hz y z-scoreada por derivacion. Un
servicio que preprocese aunque sea un poco distinto devuelve un numero al que esa medicion
ya no aplica, y no hay forma de enterarse mirando la salida. Por eso la ventana no se
reimplementa ni se copia: vive aca, sin mas dependencias que numpy y scipy, y la importan
tanto `preprocess.py` (corpus completo, Fase 2) como `inferencia.py` (un ECG suelto).

La logica no cambio al moverse de `preprocess.py`; el por que de cada paso -- por que 7,0 s,
por que se descarta en vez de rellenar con ceros, por que la ventana va centrada, por que
el z-score es por registro y por derivacion -- esta argumentado en el docstring de
`preprocess.py` y en FASES.md, Fase 2.
"""
import numpy as np
from scipy.signal import resample_poly

OUT_FREQ = 400
WINDOW_SEC = 7.0
WINDOW_SAMPLES = int(WINDOW_SEC * OUT_FREQ)  # 2800


def recortar_padding(señal: np.ndarray) -> np.ndarray:
    """Descuenta los tramos contiguos del inicio y del final donde las 12 derivaciones son
    exactamente 0 (el relleno de la fuente; ver notebooks/04_calidad_senal.ipynb)."""
    ceros = np.all(señal == 0, axis=1)
    n = len(ceros)
    inicio = 0
    while inicio < n and ceros[inicio]:
        inicio += 1
    fin = 0
    while fin < n - inicio and ceros[n - 1 - fin]:
        fin += 1
    return señal[inicio : n - fin] if fin > 0 else señal[inicio:]


def procesar_registro(señal: np.ndarray, frecuencia_nativa: int) -> tuple[np.ndarray | None, str]:
    """Devuelve (ventana normalizada (2800, 12), "ok") o (None, motivo del descarte).

    Motivos posibles: "corto" (menos de 7,0 s de señal real despues de descontar el
    padding) y "corrupto" (NaN o inf en la señal de origen).
    """
    tramo = recortar_padding(señal)
    if tramo.shape[0] / frecuencia_nativa < WINDOW_SEC:
        return None, "corto"

    # Se chequea sobre el tramo entero y antes de resamplear: resample_poly es una
    # convolucion, asi que un solo NaN fuera de la ventana se esparciria hacia adentro.
    if not np.isfinite(tramo).all():
        return None, "corrupto"

    if frecuencia_nativa != OUT_FREQ:
        g = np.gcd(int(frecuencia_nativa), OUT_FREQ)
        tramo = resample_poly(tramo, up=OUT_FREQ // g, down=int(frecuencia_nativa) // g, axis=0)

    n = tramo.shape[0]
    if n < WINDOW_SAMPLES:
        return None, "corto"  # margen ante redondeo del resampleo, no deberia pasar

    inicio = (n - WINDOW_SAMPLES) // 2
    ventana = tramo[inicio : inicio + WINDOW_SAMPLES].astype(np.float32)

    media = ventana.mean(axis=0)
    desvio = ventana.std(axis=0)
    desvio[desvio < 1e-8] = 1.0  # derivacion plana: no dividir por 0, dejarla sin escalar
    return (ventana - media) / desvio, "ok"
