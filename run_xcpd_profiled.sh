#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run XCP-D in Docker with node-level profiling, then summarize callback metrics.

Usage:
  run_xcpd_profiled.sh \
    --fmri-dir /path/to/fmriprep_derivatives \
    --output-dir /path/to/xcpd_output \
    --work-dir /path/to/work \
    --participant-label 01 [--participant-label 02 ...] \
    [--image pennlinc/xcp_d:latest] \
    [--summary-top 50] \
    [-- <additional xcp_d args>]

Required:
  --fmri-dir            Host path to preprocessing derivatives input directory.
  --output-dir          Host path for XCP-D outputs.
  --work-dir            Host path for XCP-D working directory.
  --participant-label   Participant label (repeatable).

Optional:
  --image               Docker image tag (default: pennlinc/xcp_d:latest).
  --summary-top         Keep top N nodes by runtime-estimated memory delta.
  --summary-file        Output TSV name in output/logs (default: resource_monitor_memory-summary.tsv).
  --help                Show this help text.

Notes:
  - This script always enables --resource-monitor.
  - Additional arguments after '--' are passed directly to xcp_d.
EOF
}

IMAGE="pennlinc/xcp_d:latest"
FMRI_DIR=""
OUTPUT_DIR=""
WORK_DIR=""
SUMMARY_TOP=""
SUMMARY_FILE="resource_monitor_memory-summary.tsv"
PARTICIPANTS=()
XCPD_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fmri-dir)
      FMRI_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --participant-label)
      PARTICIPANTS+=("$2")
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --summary-top)
      SUMMARY_TOP="$2"
      shift 2
      ;;
    --summary-file)
      SUMMARY_FILE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      XCPD_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$FMRI_DIR" || -z "$OUTPUT_DIR" || -z "$WORK_DIR" ]]; then
  echo "Error: --fmri-dir, --output-dir, and --work-dir are required." >&2
  usage
  exit 2
fi

if [[ ${#PARTICIPANTS[@]} -eq 0 ]]; then
  echo "Error: at least one --participant-label is required." >&2
  usage
  exit 2
fi

if [[ ! -d "$FMRI_DIR" ]]; then
  echo "Error: --fmri-dir does not exist: $FMRI_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"

XCPD_ARGS=(
  /data
  /out
  participant
  --work-dir /work
  --resource-monitor
)

for sub in "${PARTICIPANTS[@]}"; do
  XCPD_ARGS+=(--participant-label "$sub")
done

if [[ ${#XCPD_EXTRA_ARGS[@]} -gt 0 ]]; then
  XCPD_ARGS+=("${XCPD_EXTRA_ARGS[@]}")
fi

echo "Running XCP-D with profiling enabled..."
docker run --rm \
  -v "${FMRI_DIR}:/data:ro" \
  -v "${OUTPUT_DIR}:/out" \
  -v "${WORK_DIR}:/work" \
  "$IMAGE" \
  "${XCPD_ARGS[@]}"

CALLBACK_LOG_HOST="${OUTPUT_DIR}/logs/resource_monitor.jsonl"
if [[ ! -f "$CALLBACK_LOG_HOST" ]]; then
  echo "Error: callback log not found: $CALLBACK_LOG_HOST" >&2
  echo "XCP-D finished, but profiling callback output was not generated." >&2
  exit 1
fi

PROFILE_ARGS=(
  /out/logs/resource_monitor.jsonl
  --output "/out/logs/${SUMMARY_FILE}"
)

if [[ -n "$SUMMARY_TOP" ]]; then
  PROFILE_ARGS+=(--top "$SUMMARY_TOP")
fi

echo "Summarizing node-level memory usage..."
docker run --rm \
  --entrypoint /usr/local/miniconda/bin/xcp_d-profile-resources \
  -v "${OUTPUT_DIR}:/out" \
  "$IMAGE" \
  "${PROFILE_ARGS[@]}"

echo "Done."
echo "Callback log: ${CALLBACK_LOG_HOST}"
echo "Summary TSV:  ${OUTPUT_DIR}/logs/${SUMMARY_FILE}"
