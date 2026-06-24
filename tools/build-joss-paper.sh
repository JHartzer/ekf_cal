#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$repo_root/docs/paper"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found in PATH" >&2
    exit 1
fi

if [ ! -d "$paper_dir" ]; then
    echo "paper directory not found at '$paper_dir'" >&2
    exit 1
fi

docker run \
    --rm \
    --volume "$paper_dir:/data" \
    --user "$(id -u):$(id -g)" \
    --env JOURNAL=joss \
    openjournals/inara
