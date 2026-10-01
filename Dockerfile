# Imagen del servicio de inferencia (src/servidor.py). Solo CPU: ~500 ms por analisis, no
# hace falta GPU para servir.
#
#   docker build -t deca-inferencia .
#   docker run -p 8000:8000 -e DECA_API_TOKEN=<secreto> deca-inferencia
#
# Sin DECA_API_TOKEN el servicio arranca pero responde 500 a todo lo que no sea /salud: no
# hay forma de publicarlo abierto por olvido.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Las versiones salen de requirements.txt y requirements-api.txt, no se repiten aca: una
# sola fuente de pins para lo que entrena y lo que sirve. Del requirements.txt se toma solo
# lo que el servicio importa (sin h5py, pyarrow, matplotlib, jupyter ni sklearn). torch va
# desde el indice CPU de PyTorch: la rueda de PyPI trae ~2 GB de CUDA que aca no se usan.
COPY requirements.txt requirements-api.txt ./
RUN grep -E '^torch==' requirements.txt > /tmp/torch.txt \
 && grep -E '^(numpy|scipy|pandas|wfdb)==' requirements.txt > /tmp/base.txt \
 && test "$(wc -l < /tmp/base.txt)" -eq 4 \
 && pip install -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/base.txt -r requirements-api.txt

# Solo los modulos del servicio y el checkpoint congelado con su calibracion. inferencia.py
# los busca en RAIZ/models/..., con RAIZ = padre de src/, asi que se respeta esa estructura.
COPY src/servidor.py src/inferencia.py src/lectura_ecg.py src/plausibilidad.py \
     src/ventana.py src/model.py ./src/
COPY models/patrones-lr8/mejor.pt models/patrones-lr8/calibracion.json ./models/patrones-lr8/

RUN useradd --create-home --uid 1000 deca
USER deca

WORKDIR /app/src
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/salud', timeout=4)"

# Forma shell para que se expanda $PORT, que lo fijan Render, Railway, Cloud Run, etc.
# exec para que uvicorn sea PID 1 y reciba el SIGTERM del orquestador.
CMD exec uvicorn servidor:app --host 0.0.0.0 --port "$PORT" --workers 1
