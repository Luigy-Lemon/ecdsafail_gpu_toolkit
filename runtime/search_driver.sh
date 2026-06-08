#!/usr/bin/env bash
# Runs ON the GPU machine. Multi-GPU parallel chunked island search over a nonce range.
# Splits [START, START+COUNT) across all (or GPUS) visible GPUs, one process per GPU,
# each pinned via CUDA_VISIBLE_DEVICES. Emits "CLEAN nonce=N" lines (merged, deduped).
#   env:  GPU_ISLAND_BIN, GPU_STATE_FILE, BLOCKS (opt)
#   args: START COUNT [CHUNK] [NGPU=auto]
set -uo pipefail
export PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin"
START="${1:?START}"; COUNT="${2:?COUNT}"; CHUNK="${3:-200000}"; NGPU="${4:-auto}"
BIN="${GPU_ISLAND_BIN:?set GPU_ISLAND_BIN}"; STATE="${GPU_STATE_FILE:?set GPU_STATE_FILE}"; BLOCKS="${BLOCKS:-512}"
[ -x "$BIN" ] || { echo "ERROR: kernel binary not found/executable: $BIN (run build)" >&2; exit 1; }
[ -f "$STATE" ] || { echo "ERROR: state file not found: $STATE" >&2; exit 1; }
# Kernel env: KERNEL3=1 (batch-inv) or KERNEL2=1 (original shot-parallel) or neither (serial)
KFLAG=""
[ "${KERNEL3:-0}" = 1 ] && KFLAG="KERNEL3=1"
[ "${KERNEL2:-0}" = 1 ] && KFLAG="KERNEL2=1"
if [ "$NGPU" = auto ] || [ -z "$NGPU" ]; then
  unset CUDA_VISIBLE_DEVICES
else
  if [[ "$NGPU" =~ ^[0-9]+$ ]]; then
    devs=""
    for ((i=0; i<NGPU; i++)); do
      devs="${devs}${devs:+,}$i"
    done
    export CUDA_VISIBLE_DEVICES="$devs"
  else
    export CUDA_VISIBLE_DEVICES="$NGPU"
  fi
fi

env GPU_STATE="$STATE" $KFLAG BLOCKS="$BLOCKS" CHUNK="$CHUNK" \
  "$BIN" "$START" "$COUNT" | grep -oE "CLEAN nonce=[0-9]+"

