"""Valida `entrada.py` contra el codigo original de HuBERT-ECG, sobre registros PTB-XL crudos.

Dos preguntas distintas:

  (a) Implementacion: misma ventana de Fase 2 (7 s a 400 Hz) por el pipeline original y por
      `a_entrada_hubert`. Tiene que dar lo mismo salvo redondeo; si no, hay un bug nuestro.

  (b) Fidelidad: el registro crudo (10 s a 500 Hz) por el pipeline original completo, que
      es lo que HuBERT vio en el preentrenamiento, contra el mismo registro pasado por
      Fase 2 (`src/ventana.py`) y despues por `a_entrada_hubert`, sobre los mismos 5 s.
      Aca NO se espera igualdad: hay un resampleo de ida y vuelta 500->400->500 y el
      min-max se calcula sobre 7 s en vez de 10 s. Se mide cuanto se separan.

El pipeline original se ejecuta tal cual: las funciones `apply_filter`, `scaling` y
`ecg_preprocessing` se extraen del `hubert_ecg/utils.py` del repo clonado (no se copian a
mano), y `filter_signal` se carga de biosppy. El recorte + aplanado + decimate son las tres
lineas de `ECGDataset.__getitem__`. Importar los modulos enteros arrastraria torch-metrics,
loguru, peakutils, etc., por eso se carga solo lo necesario.

Los registros salen de PhysioNet (PTB-XL 1.0.3, records500, los primeros N) y no del SSD,
para que esto corra sin el disco. Se bajan aparte (el script imprime el comando si faltan).
PTB-XL usa el mismo orden de derivaciones que el proyecto.

Uso:
    python hubert/validar_entrada.py --hubert-repo <clon de Edoar-do/HuBERT-ECG> \
        --biosppy <carpeta donde esta el paquete biosppy> --datos <carpeta para los .dat> [--n 20]
"""
import argparse
import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import decimate, resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from entrada import FACTOR_DECIMADO, SEG_MUESTRAS, a_entrada_hubert  # noqa: E402
from ventana import procesar_registro  # noqa: E402

PN_URL = "https://physionet.org/files/ptb-xl/1.0.3/records500/00000"


def cargar_filter_signal(raiz_biosppy: Path):
    """Carga biosppy.signals.tools sin ejecutar el __init__ del paquete (que importa
    peakutils, matplotlib y otros que no hacen falta para filtrar)."""
    for nombre, carpeta in (("biosppy", raiz_biosppy / "biosppy"),
                            ("biosppy.signals", raiz_biosppy / "biosppy" / "signals")):
        paquete = types.ModuleType(nombre)
        paquete.__path__ = [str(carpeta)]
        sys.modules[nombre] = paquete
    for nombre, archivo in (("biosppy.utils", "biosppy/utils.py"),
                            ("biosppy.signals.tools", "biosppy/signals/tools.py")):
        spec = importlib.util.spec_from_file_location(nombre, raiz_biosppy / archivo)
        modulo = importlib.util.module_from_spec(spec)
        sys.modules[nombre] = modulo
        spec.loader.exec_module(modulo)
    return sys.modules["biosppy.signals.tools"].filter_signal


def cargar_preprocesamiento_original(repo: Path, filter_signal):
    """Extrae ecg_preprocessing (y lo que usa) del utils.py original y lo ejecuta tal cual."""
    fuente = (repo / "hubert_ecg" / "utils.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    funciones = [n for n in arbol.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("apply_filter", "scaling", "ecg_preprocessing")]
    assert len(funciones) == 3, "cambio la estructura de hubert_ecg/utils.py"
    ns = {"np": np, "filter_signal": filter_signal, "resample": resample}
    exec(compile(ast.Module(body=funciones, type_ignores=[]), "hubert_ecg/utils.py", "exec"), ns)
    return ns["ecg_preprocessing"]


def original_completo(ecg_preprocessing, señal_n12: np.ndarray, fs: int, inicio: int) -> np.ndarray:
    """ecg_preprocessing + las lineas de recorte/aplanado/decimate de ECGDataset.__getitem__."""
    ecg = ecg_preprocessing(señal_n12.T.astype(np.float64), original_frequency=fs)
    ecg = ecg[:, inicio : inicio + SEG_MUESTRAS]
    return decimate(ecg.reshape(-1), FACTOR_DECIMADO)


def comparar(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(max |x-y|, media |x-y|, correlacion minima entre derivaciones)."""
    x12, y12 = x.reshape(12, -1), y.reshape(12, -1)
    corr = min(np.corrcoef(a, b)[0, 1] for a, b in zip(x12, y12))
    return float(np.abs(x - y).max()), float(np.abs(x - y).mean()), float(corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubert-repo", type=Path, required=True)
    ap.add_argument("--biosppy", type=Path, required=True)
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    ecg_preprocessing = cargar_preprocesamiento_original(args.hubert_repo, cargar_filter_signal(args.biosppy))

    ids = [f"{i:05d}_hr" for i in range(1, args.n + 1)]
    faltan = [r for r in ids if not (args.datos / f"{r}.dat").exists()]
    if faltan:
        # No se bajan desde aca: en la red de la oficina requests falla la validacion SSL
        # (proxy con certificado propio) y curl de Windows necesita --ssl-no-revoke.
        sys.exit(f"faltan {len(faltan)} registros en {args.datos}. Bajarlos con:\n"
                 f"  for i in $(seq -f %05g 1 {args.n}); do for e in hea dat; do curl -sf "
                 f"--ssl-no-revoke -o {args.datos}/${{i}}_hr.$e {PN_URL}/${{i}}_hr.$e; done; done")

    filas_a, filas_b = [], []
    for rec in ids:
        crudo = wfdb.rdrecord(str(args.datos / rec)).p_signal  # (5000, 12) a 500 Hz
        ventana, motivo = procesar_registro(crudo, 500)  # (2800, 12) a 400 Hz, z-score
        assert ventana is not None, f"{rec}: {motivo}"

        # (a) misma ventana de 7 s, recorte central de 5 s en los dos
        inicio_7s = (3500 - SEG_MUESTRAS) // 2  # 500
        filas_a.append(comparar(original_completo(ecg_preprocessing, ventana, 400, inicio_7s),
                                a_entrada_hubert(ventana)))

        # (b) la ventana de Fase 2 arranca en la muestra 600 a 400 Hz (1,5 s): (4000-2800)//2.
        # El recorte central de 5 s dentro de ella arranca 1 s despues -> 2,5 s = muestra 1250 a 500 Hz.
        filas_b.append(comparar(original_completo(ecg_preprocessing, crudo, 500, 1250),
                                a_entrada_hubert(ventana)))

    for titulo, filas in (("(a) implementacion, misma ventana de 7 s", filas_a),
                          ("(b) fidelidad, crudo 10 s vs Fase 2 + adaptador", filas_b)):
        m = np.array(filas)
        print(f"\n{titulo}  [{len(filas)} registros; la entrada esta en [-1, 1]]")
        print(f"  max |dif|          peor registro {m[:, 0].max():.2e}   mediana {np.median(m[:, 0]):.2e}")
        print(f"  media |dif|        peor registro {m[:, 1].max():.2e}   mediana {np.median(m[:, 1]):.2e}")
        print(f"  corr min por deriv peor registro {m[:, 2].min():.6f}   mediana {np.median(m[:, 2]):.6f}")


if __name__ == "__main__":
    main()
