#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build/coverage"
coverage_info="$build_dir/coverage.info"
filtered_info="$build_dir/coverage_filtered.info"
html_dir="$build_dir/lcov"
docs_lcov_dir="$repo_root/docs/doxygen/html/lcov"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found in PATH" >&2
    exit 1
fi

if ! command -v ctest >/dev/null 2>&1; then
    echo "ctest not found in PATH" >&2
    exit 1
fi

if ! command -v lcov >/dev/null 2>&1; then
    echo "lcov not found in PATH" >&2
    exit 1
fi

if ! command -v genhtml >/dev/null 2>&1; then
    echo "genhtml not found in PATH" >&2
    exit 1
fi

rm -rf "$build_dir"

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DBUILD_WITH_ROS=OFF \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS=--coverage \
    -DCMAKE_CXX_FLAGS=--coverage

cmake --build "$build_dir" --parallel

ctest --test-dir "$build_dir" --output-on-failure

lcov \
    --directory "$build_dir" \
    --capture \
    --base-directory "$repo_root" \
    --output-file "$coverage_info" \
    --config-file "$repo_root/.lcovrc"

lcov \
    --remove "$coverage_info" \
    '*/test/*' \
    '*/usr/*' \
    '*/install/*' \
    '*/gtest/*' \
    --output-file "$filtered_info" \
    --config-file "$repo_root/.lcovrc" \
    --ignore-errors unused

genhtml \
    "$filtered_info" \
    --output-directory "$html_dir" \
    --config-file "$repo_root/.lcovrc"

rm -rf "$docs_lcov_dir"
mkdir -p "$(dirname "$docs_lcov_dir")"
cp -r "$html_dir" "$docs_lcov_dir"
