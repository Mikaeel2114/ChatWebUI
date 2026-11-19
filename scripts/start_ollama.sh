#!/usr/bin/env bash
# Starts `ollama serve`, bound to 127.0.0.1 only, with persistent RAM pinning.
set -euo pipefail
cd "$(dirname "$0")/.."

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

exec ollama serve
