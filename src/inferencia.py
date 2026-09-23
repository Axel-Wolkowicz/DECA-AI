"""Motor de inferencia: un ECG suelto -> score, banda y percentil de riesgo.

Es la pieza que traduce el checkpoint congelado a un numero que puede viajar al backend.
No abre sockets ni sabe de HTTP (eso es `servidor.py`) y no necesita el SSD: carga el
checkpoint del repo y un archivo de calibracion, y nada mas.

**Que significa el numero que devuelve, y que no.** El modelo es un **priorizador de banda
alta**, no un diagnostico ni un screener poblacional (CLAUDE.md, "SINTESIS"; FASES.md,
Fase 5). Medido sobre test el 2026-09-14: la banda alta deriva al 1,41% de la poblacion y
de esos el 29,5% resulta positivo. Por eso la salida de la sigmoide **no se reporta como
porcentaje de probabilidad**: un score de 0,93 -- que es exactamente el umbral de la banda
alta -- se leeria como "93% de chances" cuando el valor predictivo medido ahi es 29,5%.
Reportarlo asi seria errarle por un factor de tres en el sentido mas peligroso.

Lo que se reporta en su lugar es un **percentil de riesgo**: donde cae este ECG dentro de
la distribucion de scores de una poblacion de referencia. Es honesto como prioridad --
"este ECG puntua mas alto que el 98,7% de los ECG de referencia" es literalmente cierto y
es lo que el medico necesita para decidir a quien manda primero a serologia -- sin
disfrazarse de probabilidad de enfermedad.

**La poblacion de referencia es validacion, arena A (CODE-15%), a nivel paciente.** Nunca
test. Es la misma regla con la que se calibraron los umbrales (Fase 3, "Regla operativa"):
todo lo que fija un punto de operacion sale de validacion, y test se toco una sola vez
para medir. Construir el percentil sobre test gastaria una segunda mirada sin necesidad.

**El checkpoint y la calibracion van atados.** Se verifica el sha256: una calibracion
armada para otro modelo da percentiles que no significan nada, y no hay forma de notarlo
mirando la salida. Si no coinciden, esto falla en vez de responder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lectura_ecg import DERIVACIONES_CANONICAS, ECGInvalido, LecturaECG, leer_ecg
from model import ResNet1D
from ventana import OUT_FREQ, WINDOW_SEC, procesar_registro

RAIZ = Path(__file__).resolve().parent.parent
CHECKPOINT_POR_DEFECTO = RAIZ / "models" / "patrones-lr8" / "mejor.pt"
CALIBRACION_POR_DEFECTO = RAIZ / "models" / "patrones-lr8" / "calibracion.json"

# Cuantas derivaciones planas (electrodo desconectado) se toleran antes de rechazar. La
# Fase 2 no rechazaba ninguna -- deja la derivacion sin escalar y sigue -- y el modelo se
# entreno asi, con esos registros adentro. Rechazar poco es entonces lo coherente: se
# avisa siempre, y se corta recien cuando queda medio ECG o menos.
MAX_DERIVACIONES_PLANAS = 6

# Traduccion de los descartes de la Fase 2 a codigos de error del servicio. La Fase 2
# descarta en silencio porque procesa un corpus; aca hay un medico esperando una respuesta,
# asi que cada rechazo tiene que decir por que.
_MOTIVOS = {
    "corto": (
        "senal_corta",
        f"El ECG tiene menos de {WINDOW_SEC:.1f} s de señal util. El modelo se entreno "
        f"sobre ventanas de {WINDOW_SEC:.1f} s y rellenar con ceros cambiaria la señal que "
        "ve la red, asi que un registro mas corto no se puede analizar.",
    ),
    "corrupto": (
        "senal_corrupta",
        "El ECG contiene valores no finitos (NaN o infinito). Suele ser corrupcion del "
        "archivo de origen o una exportacion truncada.",
    ),
}

# Tope de duracion. El analisis usa una ventana de 7,0 s centrada, asi que una tira mas
# larga no aporta nada y un Holter de horas solo gasta CPU resampleandolo entero para
# tirar el 99,9%. 5 minutos deja pasar cualquier ECG de reposo, incluidas las tiras de
# ritmo largas, y corta lo que claramente es otra cosa.
MAX_DURACION_S = 300.0

CODIGOS = {
    "senal_corta": f"El ECG tiene menos de {WINDOW_SEC:.1f} s de señal util.",
    "senal_corrupta": "El ECG contiene valores no finitos (NaN o infinito).",
    "senal_larga": f"El registro dura mas de {MAX_DURACION_S:.0f} s; no es un ECG de reposo.",
    "senal_plana": "Demasiadas derivaciones sin señal (electrodos desconectados).",
}

ADVERTENCIA = (
    "Resultado de tamizaje: indica prioridad para la prueba serologica de Chagas, no "
    "diagnostica cardiopatia chagasica ni reemplaza la evaluacion clinica."
)


def sha256_de(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


@dataclass
class Calibracion:
    """Todo lo que fija el punto de operacion. Sale de `calibrar_servicio.py`."""

    umbral_alto: float
    grid_percentiles: np.ndarray       # scores de referencia, ordenados ascendente
    poblacion_referencia: str
    n_referencia: int
    patrones: list[str]                # orden de la cabeza de patrones del checkpoint
    calidad_cabezas: dict              # AUC/AUPRC medidos por cabeza auxiliar
    bandas: dict                       # metricas medidas de cada banda
    sha256_checkpoint: str
    generado: dict

    @classmethod
    def desde_json(cls, path: Path | str) -> "Calibracion":
        datos = json.loads(Path(path).read_text(encoding="utf-8"))
        p = datos["percentiles"]
        return cls(
            umbral_alto=float(datos["umbrales"]["umbral_alto"]),
            grid_percentiles=np.asarray(p["grid"], dtype=np.float64),
            poblacion_referencia=p["poblacion"],
            n_referencia=int(p["n"]),
            patrones=list(datos["patrones"]),
            calidad_cabezas=datos.get("calidad_cabezas", {}),
            bandas=datos["bandas"],
            sha256_checkpoint=datos["checkpoint"]["sha256"],
            generado=datos.get("generado", {}),
        )

    def percentil(self, score: float) -> float:
        """Percentil empirico del score dentro de la poblacion de referencia.

        Se usa searchsorted y no una interpolacion: el grid es la funcion de distribucion
        acumulada muestreada, y searchsorted la evalua tal cual, con los empates del lado
        correcto. La distribucion esta fuertisimamente sesgada hacia 0 (prevalencia 1,9%),
        asi que lo que importa es la resolucion arriba: con 10.001 puntos, un paso del grid
        son 3 o 4 pacientes de los ~34.800 de referencia.
        """
        g = self.grid_percentiles
        return float(np.searchsorted(g, float(score), side="right") / len(g) * 100.0)

    def banda(self, score: float) -> str:
        """`alta` o `no_alta`: el unico corte que decide una derivacion (ver
        calibrar_servicio.metricas_de_banda, decision del 2026-09-23)."""
        return "alta" if score >= self.umbral_alto else "no_alta"


class MotorDECA:
    """Checkpoint congelado + calibracion. Se construye una vez y se reusa."""

    def __init__(
        self,
        checkpoint: Path | str = CHECKPOINT_POR_DEFECTO,
        calibracion: Path | str = CALIBRACION_POR_DEFECTO,
        device: str | None = None,
        verificar_hash: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint)
        self.calibracion = Calibracion.desde_json(calibracion)
        self.sha256 = sha256_de(self.checkpoint_path)
        if verificar_hash and self.sha256 != self.calibracion.sha256_checkpoint:
            raise RuntimeError(
                "El checkpoint no es el que se calibro. Los umbrales y el percentil "
                "quedarian sin sentido y la salida no lo mostraria.\n"
                f"  checkpoint  {self.checkpoint_path}: {self.sha256[:16]}\n"
                f"  calibracion esperaba:              {self.calibracion.sha256_checkpoint[:16]}\n"
                "Recalibrar con: python src/calibrar_servicio.py --checkpoint <ruta>"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        estado = ckpt["modelo"]
        n_pat = (estado["cabeza_patrones.weight"].shape[0]
                 if "cabeza_patrones.weight" in estado else 0)
        n_demo = estado["cabeza_chagas.weight"].shape[1] - 3200
        if n_demo:
            raise NotImplementedError(
                f"El checkpoint usa {n_demo} entradas demograficas (edad/sexo) y el "
                "servicio todavia no las pide ni las normaliza. El modelo congelado del "
                "producto (patrones-lr8) se entreno sin ellas."
            )
        if n_pat != len(self.calibracion.patrones):
            raise RuntimeError(
                f"El checkpoint tiene {n_pat} cabezas de patron y la calibracion nombra "
                f"{len(self.calibracion.patrones)}: {self.calibracion.patrones}"
            )

        self.modelo = ResNet1D(n_demograficos=0, n_patrones=n_pat).to(self.device)
        self.modelo.load_state_dict(estado)
        self.modelo.eval()
        self.epoca = ckpt.get("epoca")

    # ----------------------------------------------------------------------------------
    def _preparar(self, lectura: LecturaECG) -> tuple[np.ndarray, dict]:
        """Aplica el preprocesamiento de la Fase 2 y reporta la calidad de lo que entro."""
        if lectura.duracion_s > MAX_DURACION_S:
            raise ECGInvalido(
                "senal_larga", CODIGOS["senal_larga"],
                f"{lectura.duracion_s:.0f}s recibidos; el analisis usa "
                f"{WINDOW_SEC:.1f}s centrados",
            )
        ventana, motivo = procesar_registro(lectura.señal, lectura.frecuencia)
        if ventana is None:
            codigo, mensaje = _MOTIVOS[motivo]
            raise ECGInvalido(
                codigo, mensaje,
                f"duracion recibida {lectura.duracion_s:.2f}s a {lectura.frecuencia:g} Hz",
            )

        # Una derivacion plana queda con desvio 0 despues del z-score (procesar_registro la
        # deja sin escalar en vez de dividir por cero); todas las demas quedan con desvio 1.
        planas = [DERIVACIONES_CANONICAS[i]
                  for i in np.flatnonzero(ventana.std(axis=0) < 1e-8)]
        avisos = list(lectura.avisos)
        if planas:
            avisos.append(
                f"Derivaciones sin señal (electrodo desconectado o linea plana): "
                f"{', '.join(planas)}."
            )
        if len(planas) > MAX_DERIVACIONES_PLANAS:
            raise ECGInvalido(
                "senal_plana",
                f"{len(planas)} de 12 derivaciones no tienen señal. Queda menos de medio "
                "ECG para analizar.",
                f"planas={planas}",
            )
        if lectura.derivadas and set(planas) & {"I", "II"}:
            avisos.append(
                "Atencion: se reconstruyeron derivaciones a partir de I y II, y alguna de "
                "esas dos esta plana, asi que las reconstruidas tampoco son confiables."
            )

        calidad = {
            "duracion_recibida_s": round(lectura.duracion_s, 2),
            "frecuencia_recibida_hz": lectura.frecuencia,
            "frecuencia_analizada_hz": OUT_FREQ,
            "ventana_analizada_s": WINDOW_SEC,
            "formato": lectura.formato,
            "derivaciones_recibidas": lectura.derivaciones_origen,
            "derivaciones_reconstruidas": lectura.derivadas,
            "derivaciones_planas": planas,
            "columnas_ignoradas": lectura.ignoradas,
            "avisos": avisos,
        }
        return ventana, calidad

    @torch.no_grad()
    def _puntuar(self, ventana: np.ndarray) -> tuple[float, float, np.ndarray]:
        """(2800, 12) -> probabilidades. En float32 sin autocast: con un solo registro no
        hay nada que ganar en velocidad, y la calibracion se construyo con esta misma
        precision, asi que el score y el percentil son consistentes entre si."""
        x = torch.from_numpy(np.ascontiguousarray(ventana.T, dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)
        logit_chagas, logit_rbbb, logit_patrones = self.modelo(x)
        return (
            float(torch.sigmoid(logit_chagas).item()),
            float(torch.sigmoid(logit_rbbb).item()),
            torch.sigmoid(logit_patrones).squeeze(0).cpu().numpy(),
        )

    def analizar(self, lectura: LecturaECG) -> dict:
        """La respuesta completa para un ECG. Levanta ECGInvalido si no se puede analizar."""
        ventana, calidad = self._preparar(lectura)
        score, p_rbbb, p_patrones = self._puntuar(ventana)

        cal = self.calibracion
        banda = cal.banda(score)
        percentil = cal.percentil(score)

        patrones = {"brd": self._patron("brd", p_rbbb)}
        for nombre, p in zip(cal.patrones, p_patrones):
            patrones[nombre] = self._patron(nombre, float(p))

        return {
            "percentil": round(percentil, 2),
            "banda": banda,
            "score": round(score, 6),
            "interpretacion": {**cal.bandas[banda], "advertencia": ADVERTENCIA},
            "referencia": {
                "poblacion": cal.poblacion_referencia,
                "n": cal.n_referencia,
            },
            "patrones": patrones,
            "calidad": calidad,
            "modelo": {
                "checkpoint": self.checkpoint_path.parent.name + "/" + self.checkpoint_path.name,
                "sha256": self.sha256[:16],
                "epoca": self.epoca,
                **cal.generado,
            },
        }

    def _patron(self, nombre: str, score: float) -> dict:
        """Score de una cabeza auxiliar, siempre acompañado de lo bien que mide.

        Van con su AUPRC de test al lado a proposito: `zona` mide 0,046 y es, en la
        practica, ruido. Devolverlo pelado invitaria a leerlo como un hallazgo.
        """
        calidad = self.calibracion.calidad_cabezas.get(nombre, {})
        return {"score": round(float(score), 4), **calidad}

    # ----------------------------------------------------------------------------------
    def analizar_archivo(
        self,
        datos: bytes | str | Path,
        formato: str | None = None,
        *,
        frecuencia: float | None = None,
        derivaciones: list[str] | None = None,
        nombre_archivo: str | None = None,
    ) -> dict:
        """leer + analizar, que es lo que necesita el servidor."""
        lectura = leer_ecg(datos, formato, frecuencia=frecuencia,
                           derivaciones=derivaciones, nombre_archivo=nombre_archivo)
        return self.analizar(lectura)


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="Analiza un ECG suelto con el modelo congelado")
    p.add_argument("archivo", nargs="?", help="ECG a analizar (csv/json/zip WFDB)")
    p.add_argument("--checkpoint", default=CHECKPOINT_POR_DEFECTO)
    p.add_argument("--calibracion", default=CALIBRACION_POR_DEFECTO)
    p.add_argument("--frecuencia", type=float, default=None)
    p.add_argument("--derivaciones", default=None, help="separadas por coma, si el archivo no las trae")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    t0 = time.time()
    motor = MotorDECA(args.checkpoint, args.calibracion, device=args.device)
    print(f"modelo cargado en {time.time() - t0:.1f}s sobre {motor.device}")
    print(f"  checkpoint {motor.checkpoint_path.name} sha256 {motor.sha256[:16]} epoca {motor.epoca}")
    print(f"  referencia {motor.calibracion.poblacion_referencia} "
          f"(n={motor.calibracion.n_referencia:,})")
    print(f"  umbral de la banda alta {motor.calibracion.umbral_alto:.6f}\n")

    if args.archivo:
        deriv = args.derivaciones.split(",") if args.derivaciones else None
        t0 = time.time()
        try:
            res = motor.analizar_archivo(args.archivo, frecuencia=args.frecuencia,
                                         derivaciones=deriv)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            print(f"\nanalizado en {(time.time() - t0) * 1000:.0f} ms")
        except ECGInvalido as e:
            print(f"RECHAZADO [{e.codigo}] {e.mensaje}\n  {e.detalle}")
    else:
        # Sin archivo: se puntua ruido, solo para probar que el camino entero funciona.
        ruido = LecturaECG(
            señal=np.random.default_rng(0).standard_normal((4000, 12)),
            frecuencia=500, formato="sintetico",
            derivaciones_origen=list(DERIVACIONES_CANONICAS),
        )
        res = motor.analizar(ruido)
        print("ECG sintetico (ruido blanco, no es un ECG real):")
        print(f"  score {res['score']}  percentil {res['percentil']}  banda {res['banda']}")
