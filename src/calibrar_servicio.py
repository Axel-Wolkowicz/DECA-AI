"""Construye `calibracion.json`: lo unico que el servicio necesita ademas del checkpoint.

    python src/calibrar_servicio.py --checkpoint models/patrones-lr8/mejor.pt

Necesita el SSD (corre inferencia sobre validacion). Se ejecuta una vez por checkpoint que
se quiera poner en produccion, y el archivo que sale viaja con el modelo.

**Todo lo que fija un punto de operacion sale de VALIDACION.** Los umbrales y la
distribucion de referencia del percentil se calculan sobre val / arena A (CODE-15%) a
nivel paciente -- la misma regla de la Fase 3 con la que se calibro para la medicion de
test. Test se miro una sola vez, el 2026-09-14, y calcular el percentil sobre test seria
gastar una segunda mirada para obtener algo que val da igual de bien.

**Lo unico que se toma de test son numeros ya publicados**, leidos del JSON que dejo
`evaluar_test.py`: el VPP, la sensibilidad y la fraccion de poblacion de cada banda, y la
calidad de cada cabeza auxiliar. Eso no es volver a medir test -- es la misma propiedad que
CLAUDE.md pide conservar para `analisis_errores.py --solo-analisis`: se relee un artefacto
congelado y no se retoca nada a la luz de lo que diga.

**La precision importa.** Se calibra en float32, que es lo que usa `inferencia.py`, y no en
el float16 con autocast que usa `evaluar_test.py`. Si el servicio puntuara en una precision
y el percentil viniera de otra, los dos numeros dejarian de ser comparables entre si. El
script imprime la diferencia contra los umbrales congelados para que se vea cuanto mueve.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODELOS_DIR
from dataset import PATRONES, ECGDataset, cargar_metadata_fase4, filtrar_split
from evaluar import agregar_por_paciente, calibrar_umbrales
from inferencia import CHECKPOINT_POR_DEFECTO, sha256_de
from model import ResNet1D

# Puntos del grid de percentiles. 10.001 da resolucion de 0,01 percentil: con ~34.800
# pacientes de referencia, un paso son 3 o 4 personas. Importa arriba de todo, que es
# donde vive la banda alta (el 1,4% superior) y donde la distribucion esta mas apretada.
N_GRID = 10_001

TEXTOS_BANDA = {
    "alta": (
        "Prioridad alta. Es la banda que el modelo sostiene: sobre la medicion de test, "
        "de cada 100 personas priorizadas asi cerca de 30 resultaron positivas."
    ),
    "no_alta": (
        "Sin prioridad por ECG. El trazado no muestra lo que el modelo usa para priorizar, "
        "pero esto NO descarta Chagas: cerca de 4 de cada 5 casos reales caen aca. Si la "
        "persona vive o vivio en zona endemica, su madre tiene Chagas o recibio "
        "transfusiones, la Guia nacional indica sospechar la infeccion igual, y la "
        "serologia es gratuita."
    ),
}


@torch.no_grad()
def predecir(modelo, meta, device, batch, workers) -> dict:
    """Scores por registro, alineados con `meta` por posicion (shuffle=False)."""
    loader = DataLoader(ECGDataset(meta), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    modelo.eval()
    chagas = []
    for x, *_ in tqdm(loader, desc="  inferencia val", leave=False):
        # float32, sin autocast: es la precision con la que puntua el servicio.
        lc, _, _ = modelo(x.to(device, non_blocking=True))
        chagas.append(torch.sigmoid(lc.float()).cpu().numpy())
    return np.concatenate(chagas)


def metricas_de_banda(a: dict, prevalencia: float, n: int) -> dict:
    """Las dos bandas del servicio, a partir de lo que `evaluar_test.py` ya publico.

    Es aritmetica sobre la fila `banda_alta` del JSON de test (sensibilidad, VPP y fraccion
    derivada), no una medicion nueva.

    **Dos resultados, no tres: `alta` y `no_alta`.** Decision de Axel del 2026-09-23: el
    unico uso del modelo es decidir a quien se le manda a hacer la serologia, y eso lo
    decide solo la banda alta. Hasta ese dia el servicio devolvia tambien `media` y `baja`,
    separadas por el umbral bajo de la Fase 3 (el de sensibilidad 95%), que no derivaban a
    nadie. No se las junto en una "baja": ese grupo es el 98,6% de la poblacion, tiene 1,5%
    de probabilidad contra 1,9% de la poblacion general y concentra el 78% de los casos
    reales. Llamarlo "baja" seria mentir.

    Por eso `no_alta` lleva la fraccion de casos reales que caen ahi y no un VPN: un VPN de
    98,5% suena tranquilizador, y la poblacion general ya tiene 98,1%.
    """
    pos = prevalencia * n
    alta = a["banda_alta"]
    pob_resto = 1.0 - alta["derivados"]
    perdidos = 1.0 - alta["sensibilidad"]
    return {
        "alta": {
            "texto": TEXTOS_BANDA["alta"],
            "ppv": round(float(alta["ppv"]), 4),
            "sensibilidad": round(float(alta["sensibilidad"]), 4),
            "poblacion": round(float(alta["derivados"]), 4),
        },
        "no_alta": {
            "texto": TEXTOS_BANDA["no_alta"],
            "ppv": round(float(perdidos * pos / (pob_resto * n)), 4),
            "casos_reales_perdidos": round(float(perdidos), 4),
            "poblacion": round(float(pob_resto), 4),
        },
    }


def _ultimo_test_json() -> Path | None:
    """El JSON de resultados de test mas reciente de MODELOS_DIR.

    Se filtra por contenido y no por nombre: al lado conviven otros archivos que empiezan
    con "test_" y no son resultados (por ejemplo el `.umbrales.json` que deja
    analisis_errores.py), y agarrar uno de esos daria un KeyError lejos de la causa.
    """
    candidatos = []
    for q in MODELOS_DIR.glob("test_*.json"):
        try:
            if "arenas_test" in json.loads(q.read_text(encoding="utf-8")):
                candidatos.append(q)
        except (json.JSONDecodeError, OSError):
            continue
    return max(candidatos, key=lambda q: q.stat().st_mtime, default=None)


def main():
    p = argparse.ArgumentParser(description="Calibracion del servicio de inferencia")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_POR_DEFECTO))
    p.add_argument("--test-json", default=None,
                   help="JSON de evaluar_test.py (default: el mas reciente de MODELOS_DIR)")
    p.add_argument("--out", default=None, help="default: calibracion.json junto al checkpoint")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    salida = Path(args.out) if args.out else checkpoint.with_name("calibracion.json")

    # --- lo ya medido en test: se lee, no se recalcula ---------------------------------
    ruta_test = Path(args.test_json) if args.test_json else _ultimo_test_json()
    if ruta_test is None or not ruta_test.exists():
        raise SystemExit(
            "No se encontro el JSON de evaluar_test.py. Sin el no hay de donde sacar el "
            "VPP medido de cada banda, y el servicio no puede reportar que significa el "
            "resultado. Pasarlo con --test-json."
        )
    medido = json.loads(ruta_test.read_text(encoding="utf-8"))
    if Path(medido["checkpoint"]).name != checkpoint.name or \
            Path(medido["checkpoint"]).parent.name != checkpoint.parent.name:
        print(f"AVISO: {ruta_test.name} se midio sobre {medido['checkpoint']}, "
              f"no sobre {checkpoint}")
    print(f"metricas de banda: {ruta_test.name} (test del {medido['fecha'][:10]})")

    # --- inferencia sobre validacion ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    estado = ckpt["modelo"]
    n_pat = (estado["cabeza_patrones.weight"].shape[0]
             if "cabeza_patrones.weight" in estado else 0)
    modelo = ResNet1D(n_demograficos=0, n_patrones=n_pat).to(device)
    modelo.load_state_dict(estado)
    print(f"checkpoint: {checkpoint}  (epoca {ckpt.get('epoca')}, {n_pat} cabezas de patron)")
    print(f"device: {device}, precision float32\n")

    meta_val = filtrar_split(cargar_metadata_fase4(), "val")
    scores = predecir(modelo, meta_val, device, args.batch, args.workers)

    pac = agregar_por_paciente(meta_val, scores)
    arena_a = pac[pac["dataset"] == "code15"]
    y, s = arena_a["y"].to_numpy(), arena_a["score"].to_numpy()
    # `calibrar_umbrales` devuelve tambien el umbral bajo de la Fase 3, pero el servicio ya
    # no lo usa (ver metricas_de_banda) y no se guarda: un umbral en el archivo que nadie
    # aplica invita a creer que define algo.
    umbrales = {"umbral_alto": calibrar_umbrales(y, s)["umbral_alto"]}
    congelados = {"umbral_alto": medido["umbrales_de_val"]["umbral_alto"]}

    d = umbrales["umbral_alto"] - congelados["umbral_alto"]
    print("umbral alto calibrado en val / arena A, a nivel paciente:")
    print(f"  float32 {umbrales['umbral_alto']:.9f}   congelado (fp16) "
          f"{congelados['umbral_alto']:.9f}   delta {d:+.2e}")

    # --- grid de percentiles -----------------------------------------------------------
    grid = np.quantile(s, np.linspace(0.0, 1.0, N_GRID))
    print(f"\ngrid de percentiles: {N_GRID} puntos sobre {len(s):,} pacientes de "
          f"val / arena A (CODE-15%)")
    for q in (50, 90, 99, 99.9):
        print(f"  p{q:<5} score {float(np.quantile(s, q / 100)):.6f}")

    bandas = metricas_de_banda(
        medido["arenas_test"]["arena_A"],
        medido["arenas_test"]["arena_A"]["prevalencia"],
        medido["arenas_test"]["n_pacientes"]["code15"],
    )
    print("\nbandas (metricas de test, arena A):")
    for nombre, b in bandas.items():
        detalle = (f"sens {b['sensibilidad']*100:.1f}%" if "sensibilidad" in b
                   else f"casos reales que caen aca {b['casos_reales_perdidos']*100:.1f}%")
        print(f"  {nombre:<7} {b['poblacion']*100:5.1f}% de la poblacion   "
              f"VPP {b['ppv']*100:.2f}%   {detalle}")

    calidad = {("brd" if k == "rbbb" else k): {
        "auc_test": round(v["auc"], 4), "auprc_test": round(v["auprc"], 4),
    } for k, v in medido["cabezas_test"].items()}
    # `zona` mide AUPRC 0,046: se marca para que nadie lo lea como un hallazgo.
    for nombre, v in calidad.items():
        v["usable"] = bool(v["auprc_test"] >= 0.30)

    doc = {
        "checkpoint": {"ruta": f"{checkpoint.parent.name}/{checkpoint.name}",
                       "sha256": sha256_de(checkpoint),
                       "epoca": ckpt.get("epoca")},
        "umbrales": umbrales,
        "umbrales_congelados_fp16": congelados,
        "percentiles": {
            "poblacion": "validacion / arena A (CODE-15%), a nivel paciente",
            "n": int(len(s)),
            "prevalencia": round(float(y.mean()), 6),
            "grid": [float(f"{v:.9g}") for v in grid],
        },
        "patrones": list(PATRONES),
        "calidad_cabezas": calidad,
        "bandas": bandas,
        "generado": {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "precision": "float32",
            "medicion_de_bandas": f"{ruta_test.name} (test {medido['fecha'][:10]})",
        },
    }
    salida.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida}  ({salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
