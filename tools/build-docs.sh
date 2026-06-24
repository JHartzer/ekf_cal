#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cloc_output_file="$repo_root/docs/software/cloc.md"

if ! command -v cloc >/dev/null 2>&1; then
    echo "cloc not found in PATH" >&2
    exit 1
fi

if ! command -v doxygen >/dev/null 2>&1; then
    echo "doxygen not found in PATH" >&2
    exit 1
fi

cloc_output="$(cloc "$repo_root/src" "$repo_root/eval" --md)"
cloc_table="${cloc_output#*Language|files|blank|comment|code
}"
cloc_table="${cloc_table//--------|--------|--------|--------|--------/| | | | | |}"

printf 'Count Lines of Code {#cloc}\n============\n%s' "$cloc_table" > "$cloc_output_file"

(cd "$repo_root" && doxygen .doxyfile)
