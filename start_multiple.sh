#!/usr/bin/env bash
set -e
trap 'kill 0' SIGINT

# === CONFIG ===
DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}                              # GPUs to use
BASE_PORT=${BASE_PORT:-8000}                                   # starting port number
SCRIPT=${SCRIPT:-Qwen3-VL-30B-A3B-Thinking-AWQ.bash}      # script to run
LOG_DIR=${LOG_DIR:-logs}                                   # optional log dir

# === PREPARE ===
mkdir -p "$LOG_DIR"
IFS=',' read -ra GPU_LIST <<< "$DEVICES"

# === LAUNCH ===
for idx in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$idx]}"
  PORT=$((BASE_PORT + idx))
  echo "🚀 Launching $SCRIPT on GPU ${GPU} (PORT=${PORT})"
  CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT \
    bash "$SCRIPT" > "${LOG_DIR}/gpu${GPU}.log" 2>&1 &
done

wait
