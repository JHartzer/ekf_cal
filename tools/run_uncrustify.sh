#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uncrustify >/dev/null 2>&1; then
    echo "uncrustify not found in PATH" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    exit 0
fi

for file in "$@"; do
    [ -f "$file" ] || continue
    case "$file" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx)
            uncrustify -c "$repo_root/uncrustify.cfg" --no-backup -f "$file" -o "$file"
            ;;
    esac
done
