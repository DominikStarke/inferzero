#!/usr/bin/env bash
# The container's start: the model (a catalog key, see `./zero-launcher.sh --list`, or a path to a .gguf in a mounted
# folder) downloaded into the Hugging Face cache if it is a key and not there yet, then served on all interfaces.
#   MODEL       catalog key or .gguf path (default 4b)
#   PORT        the server's port inside the container (default 8090)
#   GPUS        GPUs to split the model over (default 1)
#   SERVER_ARGS more serverz options, e.g. "--ctx 32768"
set -eu
cd /opt/inferzero
./zero-launcher.sh --check || true
# (--host 0.0.0.0: reachable through the published port; compose.yaml publishes it on 127.0.0.1 only, since the
#  server has no password of its own. --yes: the download without a question.)
# shellcheck disable=SC2086
exec ./zero-launcher.sh --model "${MODEL:-4b}" --host 0.0.0.0 --port "${PORT:-8090}" --gpus "${GPUS:-1}" \
  --no-browser --yes -- ${SERVER_ARGS:-}
