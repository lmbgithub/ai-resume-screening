#!/usr/bin/env bash
#
# Start the stack on the fastest inference backend this machine has.
#
# The order is not a preference, it is a measured one. Docker Desktop on macOS
# runs a Linux VM with no Metal passthrough, so ollama inside a container is
# CPU-only: 3.5 tok/s here against 99.7 tok/s on the host, a 28x difference
# that turns a 20-second screening run into ten minutes. So a GPU-backed
# ollama already running on the host beats a containerised one every time,
# and the containerised one is the fallback rather than the default.
#
#   ./start.sh              pick automatically
#   ./start.sh host         force the host's ollama
#   ./start.sh gpu          force the bundled ollama with NVIDIA passthrough
#   ./start.sh cpu          force the bundled ollama on CPU
#   ./start.sh --down       stop everything
set -euo pipefail

cd "$(dirname "$0")"

CHAT_MODEL="${CHAT_MODEL:-llama3.2:3b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
HOST_OLLAMA="${HOST_OLLAMA:-http://localhost:11434}"

say() { printf '  %s\n' "$*"; }

if [[ "${1:-}" == "--down" ]]; then
  docker compose --profile bundled down
  exit 0
fi

host_ollama_up() {
  curl -fsS -m 3 "$HOST_OLLAMA/api/version" >/dev/null 2>&1
}

nvidia_ready() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi -L >/dev/null 2>&1 &&
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

# --- choose the backend ------------------------------------------------------
MODE="${1:-auto}"
if [[ "$MODE" == "auto" ]]; then
  if host_ollama_up; then
    MODE=host
  elif nvidia_ready; then
    MODE=gpu
  else
    MODE=cpu
  fi
fi

echo "resume-screen — starting"

case "$MODE" in
  host)
    if ! host_ollama_up; then
      echo "error: no ollama answering at $HOST_OLLAMA. Run 'ollama serve', or use ./start.sh cpu" >&2
      exit 1
    fi
    ACCEL="$(uname -s)"
    if [[ "$ACCEL" == "Darwin" ]]; then
      say "backend: ollama on the host (Metal GPU)"
    else
      say "backend: ollama on the host"
    fi

    # The host holds the models, so pull them there rather than into a volume
    # the API will never read.
    if command -v ollama >/dev/null 2>&1; then
      installed="$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')"
      for model in "$EMBED_MODEL" "$CHAT_MODEL"; do
        # `ollama list` prints an explicit tag, so a bare name never matches
        # what is actually installed: "nomic-embed-text" is listed as
        # "nomic-embed-text:latest".
        tagged="$model"
        [[ "$tagged" == *:* ]] || tagged="$model:latest"
        if ! printf '%s\n' "$installed" | grep -qx "$tagged"; then
          say "pulling $model"
          ollama pull "$model"
        else
          say "$tagged already present"
        fi
      done
    fi

    export OLLAMA_HOST="http://host.docker.internal:11434"
    COMPOSE=(docker compose)
    ;;

  gpu)
    if ! nvidia_ready; then
      echo "error: no usable NVIDIA runtime. Use ./start.sh cpu, or ./start.sh host" >&2
      exit 1
    fi
    say "backend: bundled ollama with NVIDIA GPU passthrough"
    export OLLAMA_HOST="http://ollama:11434"
    COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile bundled)
    ;;

  cpu)
    say "backend: bundled ollama on CPU"
    say "warning: CPU inference is roughly 25-30x slower than a GPU. One"
    say "         screening run can take several minutes. This is not a hang;"
    say "         'docker compose logs -f api' prints progress per requirement."
    export OLLAMA_HOST="http://ollama:11434"
    COMPOSE=(docker compose --profile bundled)
    ;;

  *)
    echo "usage: ./start.sh [host|gpu|cpu|--down]" >&2
    exit 2
    ;;
esac

say "chat model:  $CHAT_MODEL"
say "embed model: $EMBED_MODEL"
echo

"${COMPOSE[@]}" up -d --build

echo
say "web  http://localhost:3000"
say "api  http://localhost:8000/health"
say "logs docker compose logs -f api"
