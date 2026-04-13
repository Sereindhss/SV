#!/usr/bin/env bash
# Run cProfile on the first N pairs (--profile_pairs) and save full output.
# Plan item: --profile_pairs 100; inspect damgard_jurik_reduce / pow / load.
#
# Usage (from repo root):
#   bash scripts/profile_sv_dj_match.sh [N pairs, default 100]
# Env:
#   SV_DJ_FOLDER   enrolled templates (default results/sv_dj/lfw)
#   SV_DJ_PAIR     pair list (default data/lfw/pair.list)
#   SV_DJ_FEAT     feat list (default data/lfw/lfw_feat.list)
#   SV_DJ_KS SV_DJ_K SV_DJ_S  key params (defaults 512 64 1)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

N="${1:-100}"
FOLDER="${SV_DJ_FOLDER:-results/sv_dj/lfw}"
PAIR="${SV_DJ_PAIR:-data/lfw/pair.list}"
FEAT="${SV_DJ_FEAT:-data/lfw/lfw_feat.list}"
KS="${SV_DJ_KS:-512}"
K="${SV_DJ_K:-64}"
S="${SV_DJ_S:-1}"

OUT_DIR="${REPO}/results/sv_dj"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/profile_match_${N}.log"

echo "Logging to $OUT"
{
  echo "=== SV_DJ profile_pairs=${N} $(date -Iseconds) ==="
  echo "folder=$FOLDER pair_list=$PAIR feat_list=$FEAT key_size=$KS K=$K s=$S"
  echo
} | tee "$OUT"

python3 libs/SV_DJ/crypto_system.py \
  --folder "$FOLDER" \
  --pair_list "$PAIR" \
  --feat_list "$FEAT" \
  --key_size "$KS" \
  --K "$K" \
  --s "$S" \
  --profile_pairs "$N" \
  2>&1 | tee -a "$OUT"

echo "Done. Summary: look for damgard_jurik_reduce, pow, load (numpy), decrypt in $OUT"
