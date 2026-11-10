#!/bin/bash
# =============================================================================
# run_pipeline.sh — Master pipeline for transcript extraction
#
# Steps:
#   1. load_transcripts.py        — walk HPC tree, write tidy CSV
#
# Usage:
#   sbatch run_pipeline.sh --session BW
#   sbatch run_pipeline.sh --session PMR
#   sbatch run_pipeline.sh --session all --filter-quality
#   sbatch run_pipeline.sh --session BW --person-id XXXXX
#
# Ad-hoc test run on a copied sample (alias --input for --data-root):
#   sbatch run_pipeline.sh --session BW --input /scratch/$USER/sample \
#       --output-dir /scratch/$USER/sample/out
#
# =============================================================================

#SBATCH --job-name=pipeline
#SBATCH --partition=16GB
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=./logs/pipeline_transcript_extraction_%j.out
#SBATCH --error=./logs/pipeline_transcript_extraction_%j.err

set -euo pipefail

# ── Conda env activation ─────────────────────────────────────────────────────
CONDA_INIT="/cs/opt/env/python/x86_64/miniforge/bin/activate"
CONDA_ENV="preprocess"
# shellcheck disable=SC1090
source "$CONDA_INIT" "$CONDA_ENV"

# ── Argument parsing ──────────────────────────────────────────────────────────
SESSION=""
PERSON_ID=""
FILTER_QUALITY=""
DATA_ROOT="" # Projectroot
INPUT=""
OUTPUT_DIR="data"

while [[ $# -gt 0 ]]; do
    case $1 in
        --session)         SESSION="$2"; shift 2 ;;
        --person-id)       PERSON_ID="$2"; shift 2 ;;
        --data-root)       DATA_ROOT="$2"; shift 2 ;;
        --input)           INPUT="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --filter-quality)  FILTER_QUALITY="--filter-quality"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --input wins over --data-root; both empty → fall back to HPC default already in DATA_ROOT
if [[ -n "$INPUT" ]]; then
    DATA_ROOT="$INPUT"
fi

if [[ -z "$SESSION" ]]; then
    echo "ERROR: --session is required (BW | PMR | all)"
    exit 1
fi
if [[ "$SESSION" != "BW" && "$SESSION" != "PMR" && "$SESSION" != "all" ]]; then
    echo "ERROR: --session must be 'BW', 'PMR' or 'all' (got: '$SESSION')"
    exit 1
fi

# Under SLURM, the wrapper is copied into /var/spool/slurmd/job<jobid>/ so
# BASH_SOURCE points there — useless for finding our siblings. Prefer
# $SLURM_SUBMIT_DIR (set by sbatch) and only fall back to BASH_SOURCE for
# direct local debugging without SLURM.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

mkdir -p logs "$OUTPUT_DIR"

PERSON_FLAG=""
if [[ -n "$PERSON_ID" ]]; then
    PERSON_FLAG="--person-id $PERSON_ID"
fi

TRANSCRIPTS_CSV="${OUTPUT_DIR}/transcripts_${SESSION}.csv"

# ── Step 1: load transcripts ──────────────────────────────────────────────────
echo "[$(date)] Step 1: load_transcripts.py"
python load_transcripts.py \
    --data-root "$DATA_ROOT" \
    --session-type "$SESSION" \
    --output "$TRANSCRIPTS_CSV" \
    $PERSON_FLAG $FILTER_QUALITY

echo "[$(date)] DONE. Output:"
echo "    $TRANSCRIPTS_CSV"
