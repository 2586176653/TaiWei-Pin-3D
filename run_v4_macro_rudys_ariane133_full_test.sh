#!/usr/bin/env bash
set -euo pipefail

cd /TaiWei/Beta/TaiWei-Pin-3D
source env.sh

PLATFORM=${PLATFORM:-nangate45_3D}
DESIGN=${DESIGN:-ariane133}

# v4 fixes DEF/partition instance-name canonicalization, so ariane133 macros can be matched and moved.
SCRIPT=${SCRIPT:-./scripts_openroad/refine_partition_macro_rudys_v4.py}
PYTHON=${PYTHON:-/home/ny/miniconda3/bin/python}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

VARIANT=${VARIANT:-v4_macro_rudys}
SRC=${SRC:-results/${PLATFORM}/${DESIGN}/openroad}
SEED=${SEED:-results/${PLATFORM}/${DESIGN}/openroad_v4_seed}

NET_DEF=${NET_DEF:-${SRC}/2_4_floorplan_io.def}
UPPER_DEF=${UPPER_DEF:-${SRC}/2_5_place_macro_upper.def}
BOTTOM_DEF=${BOTTOM_DEF:-${SRC}/2_5_place_macro_bottom.def}

OUT=${OUT:-${SRC}/partition.macro_rudys_v4.refined.txt}
REPORT=${REPORT:-${SRC}/partition.macro_rudys_v4.report.txt}

# Tunable weights. Override from shell without editing this file.
NET_WEIGHT_MACRO_MACRO=${NET_WEIGHT_MACRO_MACRO:-1.0}
NET_WEIGHT_MACRO_IO=${NET_WEIGHT_MACRO_IO:-1.0}
NET_WEIGHT_MACRO_STDCELL=${NET_WEIGHT_MACRO_STDCELL:-0.3}
W_RUDYS_OVERFLOW=${W_RUDYS_OVERFLOW:-0.2}
W_RUDYS_TOPK=${W_RUDYS_TOPK:-0.1}
W_CUT=${W_CUT:-100.0}
W_AREA_BALANCE=${W_AREA_BALANCE:-0.0}
MAX_MACRO_BALANCE=${MAX_MACRO_BALANCE:-0.75}
NUM_CORES=${NUM_CORES:-1}
OUTER_ITERATIONS=${OUTER_ITERATIONS:-1}

if [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] Missing script: $SCRIPT"
  exit 1
fi

for def_file in "$NET_DEF" "$UPPER_DEF" "$BOTTOM_DEF"; do
  if [ ! -f "$def_file" ]; then
    echo "[ERROR] Missing DEF: $def_file"
    exit 1
  fi
done

if [ -f "${SRC}/partition.txt.before_v2_true_macro_pins" ]; then
  PART=${PART:-${SRC}/partition.txt.before_v2_true_macro_pins}
else
  PART=${PART:-${SRC}/partition.txt}
fi

if [ ! -f "$PART" ]; then
  echo "[ERROR] Missing input partition: $PART"
  exit 1
fi

mapfile -t MACRO_LEFS < <(find \
  "platforms/${PLATFORM}/lef_bottom/fakeram_block" \
  "platforms/${PLATFORM}/lef_upper/fakeram_block" \
  -name "*.lef" | sort)

if [ "${#MACRO_LEFS[@]}" -eq 0 ]; then
  echo "[ERROR] No macro LEFs found for platform: $PLATFORM"
  exit 1
fi

echo "[STEP 1] Run v4 macro-aware Rudys refinement for ${PLATFORM}/${DESIGN}"
"$PYTHON" "$SCRIPT" refine \
  --net-def "$NET_DEF" \
  --upper-def "$UPPER_DEF" \
  --bottom-def "$BOTTOM_DEF" \
  --partition-in "$PART" \
  --partition-out "$OUT" \
  --report "$REPORT" \
  --lef "${MACRO_LEFS[@]}" \
  --pin-location-mode macro \
  --rudy-net-scope macro_related \
  --net-weight-macro-macro "$NET_WEIGHT_MACRO_MACRO" \
  --net-weight-macro-io "$NET_WEIGHT_MACRO_IO" \
  --net-weight-macro-stdcell "$NET_WEIGHT_MACRO_STDCELL" \
  --score-normalization initial \
  --w-rudys-overflow "$W_RUDYS_OVERFLOW" \
  --w-rudys-topk "$W_RUDYS_TOPK" \
  --w-cut "$W_CUT" \
  --w-area-balance "$W_AREA_BALANCE" \
  --max-macro-balance "$MAX_MACRO_BALANCE"

echo "[STEP 2] Check v4 report"
grep -E "num_macros|num_movable_macros|rudy_net_scope|net_class_|macro_pin_geometry_coverage|initial_score|final_score|cut_nets|normalized_cut|macro_area_balance|moved_macros" "$REPORT" || true

echo "[STEP 3] Create v4 seed variant"
rm -rf "$SEED"
mkdir -p "$SEED"

cp -f "$SRC/1_synth.sdc" "$SEED/"
cp -f "$NET_DEF" "$SEED/"
cp -f "$UPPER_DEF" "$SEED/"
cp -f "$BOTTOM_DEF" "$SEED/"
cp -f "$SRC/2_2_floorplan_io.def" "$SEED/" 2>/dev/null || true
cp -f "$SRC/2_2_floorplan_io.v" "$SEED/" 2>/dev/null || true
cp -f "$SRC/1_synth.v" "$SEED/" 2>/dev/null || true
cp -f "$SRC/partition.result.tcl" "$SEED/" 2>/dev/null || true
cp -f "$SRC/partition.simple_plan.txt" "$SEED/" 2>/dev/null || true
cp -f "$OUT" "$SEED/partition.txt"

cmp -s "$PART" "$SEED/partition.txt" \
  && echo "[WARN] v4 partition unchanged from input" \
  || echo "[INFO] OK: seed partition is v4 refined"

echo "[STEP 4] Run official OpenROAD 3D flow with v4 partition"
rm -rf \
  "results/${PLATFORM}/${DESIGN}/${VARIANT}" \
  "logs/${PLATFORM}/${DESIGN}/${VARIANT}" \
  "reports/${PLATFORM}/${DESIGN}/${VARIANT}" \
  "objects/${PLATFORM}/${DESIGN}/${VARIANT}"

REUSE_2DPART_FROM_VARIANT=openroad_v4_seed \
OUTER_ITERATIONS="$OUTER_ITERATIONS" \
NUM_CORES="$NUM_CORES" \
bash test/openroad/ORD_3D_NEW_FLOW.sh \
  "$PLATFORM" \
  "$VARIANT" \
  openroad \
  "$DESIGN" 2>&1 | tee "run_${DESIGN}_${VARIANT}_official.log"

echo "[STEP 5] Run official Cadence/Innovus eval"
bash test/common/run_stage.sh \
  "$PLATFORM" \
  "$VARIANT" \
  openroad \
  "$DESIGN" \
  cds-final 2>&1 | tee "eval_${DESIGN}_${VARIANT}_cds_final.log"

echo "[STEP 6] Show final summary"
SUMMARY="logs/${PLATFORM}/${DESIGN}/${VARIANT}/final_summary.txt"

if [ -f "$SUMMARY" ]; then
  cat "$SUMMARY"
else
  echo "[ERROR] Missing final summary: $SUMMARY"
  exit 1
fi

echo "[STEP 7] Compare baseline vs v4"
BASE="logs/${PLATFORM}/${DESIGN}/openroad/final_summary.txt"

if [ -f "$BASE" ]; then
  diff -u "$BASE" "$SUMMARY" | sed -n '1,240p' || true
else
  echo "[WARN] Missing baseline summary: $BASE"
fi

echo "[DONE] v4 full test completed for ${PLATFORM}/${DESIGN}."
