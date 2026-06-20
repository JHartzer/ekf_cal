#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
perf_dir="$script_dir/perf"
sim_binary="$repo_root/build/RelWithDebInfo/sim"

if [ ! -x "$sim_binary" ]; then
    echo "sim binary not found at '$sim_binary'" >&2
    echo "Build it first with: cmake --build --preset RelWithDebInfo" >&2
    exit 1
fi

mkdir -p "$perf_dir"

"$sim_binary" perf.yaml "$perf_dir/" &
sim_pid=$!
trap 'kill "$sim_pid" 2>/dev/null || true' EXIT

perf record -F 99 -p "$sim_pid" --call-graph dwarf -- sleep 60
perf script > "$script_dir/sim.perf"
"$script_dir/stackcollapse-perf.pl" "$script_dir/sim.perf" > "$script_dir/sim.folded"
"$script_dir/flamegraph.pl" "$script_dir/sim.folded" > "$script_dir/flamegraph.svg"
