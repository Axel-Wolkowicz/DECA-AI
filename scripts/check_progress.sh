#!/usr/bin/env bash
# Muestra el progreso de una conversion que loguea con tqdm (ej. convert_ptbxl.py).
#
# tqdm escribe su barra con \r (sobreescribe la misma linea en una terminal real),
# pero al redirigir a un archivo esos \r quedan como texto: hay que separarlos por
# \n y quedarse con el ultimo para ver el estado actual.
#
# Uso:
#   scripts/check_progress.sh <log_file>            # una foto del progreso actual
#   scripts/check_progress.sh <log_file> --watch     # se refresca cada 5s (Ctrl+C para salir)
set -u

LOG="${1:?Uso: check_progress.sh <log_file> [--watch]}"
WATCH="${2:-}"

show() {
  if [ ! -f "$LOG" ]; then
    echo "No existe el log: $LOG"
    return 1
  fi
  tr '\r' '\n' < "$LOG" | grep -v '^[[:space:]]*$' | tail -1
}

if [ "$WATCH" = "--watch" ]; then
  while true; do
    clear
    echo "=== $(date '+%H:%M:%S') === $LOG"
    show
    sleep 5
  done
else
  show
fi
