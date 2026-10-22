"""Adapta una ventana de Fase 2 al formato de entrada de HuBERT-ECG.

HuBERT-ECG (Coppola et al., github.com/Edoar-do/HuBERT-ECG) se preentreno con una entrada
muy distinta a la nuestra: 5 s a 100 Hz, pasabanda 0,05-47 Hz, min-max a [-1, 1] por
derivacion y las 12 derivaciones aplanadas en un solo vector de 6000 muestras. Este modulo
replica ese pipeline partiendo de la ventana (2800, 12) a 400 Hz z-scoreada que produce
`src/ventana.py`, para no tener que volver a la señal cruda.

Que eso sea posible no es obvio y depende de dos propiedades: el min-max es invariante a
cualquier reescalado lineal positivo, y el pasabanda elimina la constante. Entonces el
z-score de Fase 2 (restar la media, dividir por el desvio) desaparece al llegar al min-max,
salvo efectos de borde. `validar_entrada.py` mide cuanto se separa esto de lo que el modelo
vio en el preentrenamiento.

Los pasos replican, en el mismo orden, `ecg_preprocessing` de hubert_ecg/utils.py y el
`__getitem__` de hubert_ecg/dataset.py:

    1. Pasabanda FIR 0,05-47 Hz a la frecuencia nativa, fase cero (filtfilt). Es lo que
       hace biosppy.signals.tools.filter_signal, reimplementado aca para no sumar biosppy
       como dependencia: numtaps = int(0,3 * fs), +1 si es par (biosppy lo fuerza impar).
    2. Resampleo a 500 Hz con scipy.signal.resample (por FFT, no resample_poly como Fase 2).
    3. Min-max a [-1, 1] por derivacion, sobre la ventana entera ANTES de recortar: ellos
       escalan el registro completo y recortan despues, en el dataset.
    4. Recorte de 5 s = 2500 muestras a 500 Hz. Centrado por defecto (ver `inicio`).
    5. Aplanado derivacion por derivacion -> (30000,) y scipy.signal.decimate(x, 5) sobre
       el vector aplanado -> (6000,). Ojo: el filtro antialiasing de decimate cruza el borde
       entre derivaciones. Es una rareza del original y se replica a proposito, porque es
       lo que el modelo vio en el preentrenamiento.

Solo depende de numpy y scipy, como `src/ventana.py`, para que el servicio de inferencia
pueda usarlo sin el corpus.
"""
import numpy as np
from scipy.signal import decimate, filtfilt, firwin, resample

FREQ_HUBERT = 500
SEG_MUESTRAS = 2500  # 5 s a 500 Hz
FACTOR_DECIMADO = 5  # 500 Hz -> 100 Hz
BANDA_HZ = (0.05, 47.0)
LARGO_ENTRADA = 12 * SEG_MUESTRAS // FACTOR_DECIMADO  # 6000


def filtrar(señal: np.ndarray, fs: float) -> np.ndarray:
    """Pasabanda FIR de fase cero sobre (12, n), identico a biosppy filter_signal(ftype="FIR")."""
    numtaps = int(0.3 * fs)
    if numtaps % 2 == 0:
        numtaps += 1
    b = firwin(numtaps, cutoff=2.0 * np.asarray(BANDA_HZ) / fs, pass_zero=False)
    return filtfilt(b, np.array([1]), señal)


def escalar(señal: np.ndarray, suave: float = 1e-8) -> np.ndarray:
    """Min-max a [-1, 1] por derivacion, misma formula (y mismo `suave`) que el original."""
    minimo = señal.min(axis=1, keepdims=True)
    maximo = señal.max(axis=1, keepdims=True)
    return 2 * (señal - minimo) / (maximo - minimo + suave) - 1


def a_entrada_hubert(ventana: np.ndarray, fs: float = 400, inicio: int | None = None) -> np.ndarray:
    """(n, 12) a `fs` Hz -> (6000,) float32 listo para HuBERT-ECG.

    `inicio` es la primera muestra del recorte de 5 s, medida a 500 Hz. Por defecto el
    recorte va centrado: con la ventana de 7 s de Fase 2 deja 1 s de margen a cada lado,
    que cubre el transitorio del filtro (121 coeficientes a 400 Hz = 0,3 s).
    """
    señal = filtrar(np.asarray(ventana, dtype=np.float64).T, fs)
    señal = resample(señal, int(señal.shape[-1] * (FREQ_HUBERT / fs)), axis=1)
    señal = escalar(señal)

    n = señal.shape[1]
    if n < SEG_MUESTRAS:
        raise ValueError(f"la señal dura {n / FREQ_HUBERT:.2f} s a 500 Hz, hacen falta 5 s")
    if inicio is None:
        inicio = (n - SEG_MUESTRAS) // 2
    señal = señal[:, inicio : inicio + SEG_MUESTRAS]

    return decimate(señal.reshape(-1), FACTOR_DECIMADO).astype(np.float32)
