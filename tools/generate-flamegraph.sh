#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
flamegraph_dir="$repo_root/docs/flamegraph"
perf_dir="$flamegraph_dir/perf"
perf_config="$flamegraph_dir/perf.yaml"
stackcollapse_script="$flamegraph_dir/stackcollapse-perf.pl"
flamegraph_script="$script_dir/flamegraph.pl"
sim_binary="$repo_root/build/RelWithDebInfo/sim"
perf_script="$flamegraph_dir/sim.perf"
folded_stacks="$flamegraph_dir/sim.folded"
output_svg="$flamegraph_dir/flamegraph.svg"

if [ ! -x "$sim_binary" ]; then
    echo "sim binary not found at '$sim_binary'" >&2
    echo "Build it first with: cmake --build --preset RelWithDebInfo" >&2
    exit 1
fi

if [ ! -x "$stackcollapse_script" ]; then
    echo "stackcollapse script not found at '$stackcollapse_script'" >&2
    exit 1
fi

if [ ! -x "$flamegraph_script" ]; then
    echo "flamegraph script not found at '$flamegraph_script'" >&2
    exit 1
fi

mkdir -p "$perf_dir"

"$sim_binary" "$perf_config" "$perf_dir/" &
sim_pid=$!
trap 'kill "$sim_pid" 2>/dev/null || true' EXIT

perf record -F 99 -p "$sim_pid" --call-graph dwarf -- sleep 60
perf script > "$perf_script"
"$stackcollapse_script" "$perf_script" > "$folded_stacks"
"$flamegraph_script" "$folded_stacks" > "$output_svg"
