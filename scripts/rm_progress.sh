#!/usr/bin/env bash
# Borra las subcarpetas de un directorio una por una, logueando el progreso.
#
# `rm -rf` sobre un directorio con miles de archivos chicos (exFAT, USB) no da
# ninguna senal de avance hasta que termina del todo. Esto borra subcarpeta por
# subcarpeta para poder ver cuanto falta con check_progress.sh mientras corre.
#
# Uso: rm_progress.sh <directorio_padre> [log_file]
set -u

DIR="${1:?Uso: rm_progress.sh <directorio_padre> [log_file]}"
LOG="${2:-$(dirname "$DIR")/$(basename "$DIR").rm.log}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ ! -d "$DIR" ]; then
  log "No existe: $DIR"
  exit 1
fi

mapfile -t SUBS < <(ls -1 "$DIR")
TOTAL=${#SUBS[@]}

log "=== borrando $TOTAL subcarpetas de $DIR ==="

i=0
for sub in "${SUBS[@]}"; do
  i=$((i + 1))
  rm -rf "${DIR:?}/${sub}"
  log "[$i/$TOTAL] borrado $sub"
done

rmdir "$DIR" 2>/dev/null

log "=== listo: $DIR eliminado ==="
