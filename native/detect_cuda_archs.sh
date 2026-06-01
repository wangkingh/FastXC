#!/usr/bin/env sh

set -u

fallback="${CUDA_ARCH_FALLBACK:-sm_80}"
check_gpu="${CHECK_GPU:-}"
probe_timeout="${CUDA_ARCH_PROBE_TIMEOUT:-3s}"

run_probe() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "$probe_timeout" "$@"
  else
    "$@"
  fi
}

normalize_archs() {
  awk '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      if ($0 == "") next
      gsub(/^sm_/, "", $0)
      gsub(/\./, "", $0)
      if ($0 ~ /^[1-9][0-9]$/) print "sm_" $0
    }
  ' | sort -u | tr '\n' ' '
}

from_check_gpu() {
  [ -n "$check_gpu" ] || return 1
  [ -x "$check_gpu" ] || return 1
  run_probe "$check_gpu" 2>/dev/null | awk -F: '/Compute capability/ {print $2}' | normalize_archs
}

from_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  run_probe nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | normalize_archs
}

archs="$(from_check_gpu)"
[ -n "$archs" ] || archs="$(from_nvidia_smi)"
[ -n "$archs" ] || archs="$fallback"

printf '%s\n' "$archs" | normalize_archs
