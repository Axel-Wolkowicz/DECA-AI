"""Entrypoint del despliegue en TIC, que ejecuta `uvicorn main:app` desde la raiz del repo.

El servicio vive en `src/servidor.py`; esto solo lo expone con el nombre que TIC espera.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from servidor import app  # noqa: E402,F401
