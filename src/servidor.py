"""El servicio HTTP que consume el backend de DECA.

    uvicorn servidor:app --host 0.0.0.0 --port 8000     (desde src/)
    python src/servidor.py --puerto 8000                (equivalente, con chequeos)

**Por que existe un servicio aparte en vez de meter el modelo en el backend.** DECA-Back
es Node sobre Vercel serverless: no puede ejecutar PyTorch ni lanzar un proceso Python. Las
alternativas eran exportar a ONNX y correrlo en Node -- el modelo solo pesa 78 MB contra
los 250 MB de bundle, y ademas obligaria a reescribir el resampleo y el z-score en
JavaScript. Eso ultimo es el problema real: **si el preprocesamiento no es identico al de
la Fase 2, las metricas medidas dejan de aplicar al numero que devuelve la API**, y no hay
forma de darse cuenta mirando la salida. Asi que la señal la procesa el mismo codigo que
proceso el corpus (`ventana.py`) y el backend habla HTTP.

**Este servicio no se expone a internet.** No tiene usuarios, ni sesiones, ni sabe quien es
el paciente: recibe una señal y devuelve un numero. Quien decide si el medico puede ver ese
paciente es el backend, que ya tiene el JWT y el chequeo de asignacion. La unica
autenticacion de aca es un secreto compartido (`DECA_API_TOKEN`) para que nadie mas que el
backend pueda gastarle GPU.

El servicio es **sin estado y sin memoria**: no guarda la señal, no la loguea y no la
escribe a disco. Lo que entra se procesa en RAM y se descarta. Guardar el ECG es decision
del backend, que es el que tiene el consentimiento y la historia clinica.
"""
from __future__ import annotations

import hmac
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

import inferencia
import lectura_ecg
from inferencia import (
    ADVERTENCIA,
    CALIBRACION_POR_DEFECTO,
    CHECKPOINT_POR_DEFECTO,
    MotorDECA,
)
from lectura_ecg import DERIVACIONES_CANONICAS, ECGInvalido, LECTORES

# Un ECG de reposo de 10 s a 500 Hz son ~600 KB en CSV y ~1,2 MB en JSON. El limite deja
# margen de sobra para eso y corta cualquier cosa que sea otra cosa.
MAX_BYTES = 32 * 1024 * 1024

TOKEN = os.environ.get("DECA_API_TOKEN", "")
SIN_AUTH = os.environ.get("DECA_API_SIN_AUTH", "").lower() in ("1", "true", "si")

motor: MotorDECA | None = None


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """El modelo se carga una vez al arrancar, no por pedido: son ~2 s de carga."""
    global motor
    motor = MotorDECA(
        os.environ.get("DECA_CHECKPOINT", CHECKPOINT_POR_DEFECTO),
        os.environ.get("DECA_CALIBRACION", CALIBRACION_POR_DEFECTO),
        device=os.environ.get("DECA_DEVICE") or None,
    )
    print(f"modelo listo en {motor.device} | checkpoint {motor.sha256[:12]} "
          f"| auth {'DESACTIVADA' if SIN_AUTH else 'activa'}")
    yield


app = FastAPI(
    title="DECA -- servicio de inferencia",
    description=(
        "Priorizador de tamizaje de cardiopatia chagasica a partir del ECG de 12 "
        "derivaciones. No diagnostica."
    ),
    version="1.0",
    lifespan=ciclo_de_vida,
)


def autenticar(x_deca_token: str = Header(default="")):
    """Secreto compartido con el backend. `compare_digest` y no `==` para no filtrar el
    largo del token por tiempo de respuesta."""
    if SIN_AUTH:
        return
    if not TOKEN:
        raise _error(500, "sin_token_configurado",
                     "El servicio arranco sin DECA_API_TOKEN y sin DECA_API_SIN_AUTH.")
    if not hmac.compare_digest(x_deca_token, TOKEN):
        raise _error(401, "token_invalido", "Falta el header X-DECA-Token o no coincide.")


class _ErrorHTTP(Exception):
    def __init__(self, status, codigo, mensaje, detalle=""):
        self.status, self.codigo, self.mensaje, self.detalle = status, codigo, mensaje, detalle


def _error(status, codigo, mensaje, detalle=""):
    return _ErrorHTTP(status, codigo, mensaje, detalle)


def _respuesta_error(status, codigo, mensaje, detalle=""):
    """Mismo sobre `{ok, error}` que ya usa DECA-Back, para que el backend no traduzca."""
    cuerpo = {"ok": False, "error": {"codigo": codigo, "mensaje": mensaje}}
    if detalle:
        cuerpo["error"]["detalle"] = detalle
    return JSONResponse(status_code=status, content=cuerpo)


@app.exception_handler(_ErrorHTTP)
async def _manejar_error_http(request: Request, exc: _ErrorHTTP):
    return _respuesta_error(exc.status, exc.codigo, exc.mensaje, exc.detalle)


@app.exception_handler(ECGInvalido)
async def _manejar_ecg_invalido(request: Request, exc: ECGInvalido):
    # 422: la peticion esta bien formada, el ECG no sirve. Es un resultado esperable del
    # servicio, no una falla -- el backend lo muestra como un mensaje, no como un error.
    return _respuesta_error(422, exc.codigo, exc.mensaje, exc.detalle)


# --------------------------------------------------------------------------------------
@app.get("/salud")
def salud():
    """Sin autenticacion: la usa el orquestador para saber si el proceso vive."""
    if motor is None:
        return _respuesta_error(503, "modelo_no_cargado", "El modelo todavia no cargo.")
    return {
        "ok": True,
        "device": str(motor.device),
        "checkpoint": motor.checkpoint_path.name,
        "sha256": motor.sha256[:16],
        "calibracion": motor.calibracion.generado,
    }


@app.get("/contrato", dependencies=[Depends(autenticar)])
def contrato():
    """Todo lo que el backend necesita saber para integrarse, servido por el servicio
    mismo: formatos, nombres de derivacion, codigos de error y que significa cada banda."""
    cal = motor.calibracion
    return {
        "ok": True,
        "formatos": sorted(LECTORES),
        "derivaciones_canonicas": list(DERIVACIONES_CANONICAS),
        "derivables_de_I_y_II": list(lectura_ecg.DERIVABLES),
        "frecuencia_obligatoria_en": ["csv"],
        "max_bytes": MAX_BYTES,
        "bandas": cal.bandas,
        "referencia_percentil": {
            "poblacion": cal.poblacion_referencia,
            "n": cal.n_referencia,
        },
        "codigos_error": {**lectura_ecg.CODIGOS, **inferencia.CODIGOS},
        "advertencia": ADVERTENCIA,
    }


async def _leer_cuerpo(archivo: UploadFile) -> bytes:
    datos = await archivo.read()
    if not datos:
        raise _error(422, "senal_vacia", "El archivo llego vacio.")
    if len(datos) > MAX_BYTES:
        raise _error(413, "archivo_demasiado_grande",
                     f"El archivo supera el limite de {MAX_BYTES // (1024*1024)} MB.",
                     f"recibidos {len(datos) // 1024} KB")
    return datos


@app.post("/analizar", dependencies=[Depends(autenticar)])
async def analizar(
    archivo: UploadFile = File(..., description="ECG en csv, json o zip WFDB"),
    frecuencia: float | None = Form(None, description="Hz; obligatorio para CSV"),
    derivaciones: str | None = Form(None, description="nombres separados por coma"),
    formato: str | None = Form(None, description="csv|json|wfdb; por defecto se detecta"),
):
    """Analiza un ECG subido como archivo. Es el camino principal."""
    datos = await _leer_cuerpo(archivo)
    nombres = [d.strip() for d in derivaciones.split(",")] if derivaciones else None
    analisis = motor.analizar_archivo(
        datos, formato, frecuencia=frecuencia, derivaciones=nombres,
        nombre_archivo=archivo.filename,
    )
    return {"ok": True, "analisis": analisis}


@app.post("/analizar/json", dependencies=[Depends(autenticar)])
async def analizar_json(request: Request):
    """Analiza un ECG mandado como JSON en el cuerpo, para un frontend que ya tiene la
    matriz en memoria y no quiere armar un archivo.

    El cuerpo va crudo al mismo lector que atiende los archivos .json, asi que las dos
    puertas validan exactamente igual. Forma esperada:
        {"frecuencia": 500, "derivaciones": ["I", ...], "senal": [[...], ...]}
    """
    datos = await request.body()
    if not datos:
        raise _error(422, "senal_vacia", "El cuerpo del pedido llego vacio.")
    if len(datos) > MAX_BYTES:
        raise _error(413, "archivo_demasiado_grande",
                     f"El cuerpo supera el limite de {MAX_BYTES // (1024*1024)} MB.")
    return {"ok": True, "analisis": motor.analizar_archivo(datos, "json")}


if __name__ == "__main__":
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Servicio de inferencia de DECA")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 por defecto: no se expone a la red sin pedirlo")
    p.add_argument("--puerto", type=int, default=8000)
    p.add_argument("--recargar", action="store_true", help="autorecarga, solo para desarrollo")
    args = p.parse_args()

    if not TOKEN and not SIN_AUTH:
        raise SystemExit(
            "Falta DECA_API_TOKEN. El servicio no arranca sin autenticacion para que no se\n"
            "publique por accidente un endpoint que cualquiera puede usar.\n"
            '  PowerShell:  $env:DECA_API_TOKEN = "<secreto>"\n'
            "  bash:        export DECA_API_TOKEN=<secreto>\n"
            "Para desarrollo local, a sabiendas:  DECA_API_SIN_AUTH=1"
        )
    if not Path(CALIBRACION_POR_DEFECTO).exists():
        raise SystemExit(
            f"Falta {CALIBRACION_POR_DEFECTO}.\n"
            "Generarla con: python src/calibrar_servicio.py --checkpoint "
            f"{CHECKPOINT_POR_DEFECTO}"
        )

    # Coherencia del catalogo de errores: que /contrato no prometa codigos que no existen
    # ni se calle los que si. Es barato y evita que la documentacion se desincronice.
    import re
    catalogo = {**lectura_ecg.CODIGOS, **inferencia.CODIGOS}
    en_codigo = {c for c, _ in inferencia._MOTIVOS.values()}
    for modulo in (lectura_ecg, inferencia):
        fuente = Path(modulo.__file__).read_text(encoding="utf-8")
        en_codigo |= set(re.findall(r'ECGInvalido\(\s*"([a-z_]+)"', fuente))
    if faltan := en_codigo - set(catalogo):
        print(f"AVISO: codigos que se lanzan pero no estan en el catalogo: {sorted(faltan)}")
    if sobran := set(catalogo) - en_codigo:
        print(f"AVISO: codigos en el catalogo que no lanza nadie: {sorted(sobran)}")

    uvicorn.run("servidor:app", host=args.host, port=args.puerto, reload=args.recargar)
