#!/usr/bin/env bash
# Descarga un archivo grande en N chunks paralelos con reintento por chunk.
#
# Uso: dl_chunked.sh <url> <salida> <bytes_esperados> <n_chunks> [offset_inicial]
#
# Cada chunk se descarga a <salida>.parts/NN y se reintenta hasta tener el
# tamano exacto que le corresponde. Recien cuando todos estan completos se
# concatenan (en orden) sobre <salida>. Si <salida> ya tiene un prefijo valido,
# pasar su tamano como offset_inicial para reanudar sin volver a bajar eso.
#
# Es idempotente: se puede volver a correr y solo baja lo que falta.
set -u

URL="$1"; OUT="$2"; EXPECTED="$3"; NCHUNKS="$4"; START="${5:-0}"
PARTS="${OUT}.parts"
LOG="$(dirname "$OUT")/$(basename "$OUT").dl.log"
MAX_TRIES=200

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }

# Ya esta completo
if [ "$(fsize "$OUT")" = "$EXPECTED" ]; then
  log "YA COMPLETO: $OUT ($EXPECTED bytes)"
  exit 0
fi

mkdir -p "$PARTS"

REMAINING=$((EXPECTED - START))
CHUNK=$((REMAINING / NCHUNKS))

log "=== $(basename "$OUT"): faltan $REMAINING bytes desde offset $START en $NCHUNKS chunks ==="

# Lanza un worker por chunk
for i in $(seq 0 $((NCHUNKS - 1))); do
  CS=$((START + i * CHUNK))
  if [ "$i" = "$((NCHUNKS - 1))" ]; then CE=$((EXPECTED - 1)); else CE=$((CS + CHUNK - 1)); fi
  NEED=$((CE - CS + 1))
  PF="$PARTS/$(printf '%02d' "$i")"
  (
    try=0
    while [ "$(fsize "$PF")" -lt "$NEED" ]; do
      try=$((try + 1))
      if [ "$try" -gt "$MAX_TRIES" ]; then
        echo "[chunk $i] AGOTADOS $MAX_TRIES intentos" >> "$LOG"; exit 1
      fi
      HAVE=$(fsize "$PF")
      # curl escribe a stdout y appendeamos: reanuda exactamente donde quedo
      curl -sS -L --fail --connect-timeout 30 --max-time 3600 \
           --retry 3 --retry-delay 5 --retry-all-errors \
           -r "$((CS + HAVE))-${CE}" "$URL" >> "$PF" 2>>"$LOG"
      NOW=$(fsize "$PF")
      if [ "$NOW" -gt "$NEED" ]; then
        # El server ignoro el Range o mando de mas: descartar y reempezar el chunk
        echo "[chunk $i] sobrante ($NOW > $NEED), reiniciando chunk" >> "$LOG"
        rm -f "$PF"
      elif [ "$NOW" = "$HAVE" ]; then
        sleep 5   # no avanzo nada, esperar antes de reintentar
      fi
    done
  ) &
done

wait

# Verificar que todos los chunks tengan el tamano exacto
TOTAL=0
for i in $(seq 0 $((NCHUNKS - 1))); do
  CS=$((START + i * CHUNK))
  if [ "$i" = "$((NCHUNKS - 1))" ]; then CE=$((EXPECTED - 1)); else CE=$((CS + CHUNK - 1)); fi
  NEED=$((CE - CS + 1))
  GOT=$(fsize "$PARTS/$(printf '%02d' "$i")")
  if [ "$GOT" != "$NEED" ]; then
    log "ERROR chunk $i incompleto (got=$GOT need=$NEED). Volver a correr el script para reanudar."
    exit 1
  fi
  TOTAL=$((TOTAL + GOT))
done

log "chunks OK ($TOTAL bytes), concatenando..."

if [ "$START" = "0" ]; then : > "$OUT"; else truncate -s "$START" "$OUT"; fi
for i in $(seq 0 $((NCHUNKS - 1))); do
  cat "$PARTS/$(printf '%02d' "$i")" >> "$OUT"
done

FINAL=$(fsize "$OUT")
if [ "$FINAL" = "$EXPECTED" ]; then
  rm -rf "$PARTS"
  log "OK $(basename "$OUT") completo ($FINAL bytes)"
  exit 0
else
  log "ERROR tamano final $FINAL != $EXPECTED"
  exit 1
fi
