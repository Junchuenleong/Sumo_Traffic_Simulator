#!/usr/bin/env bash
# =========================================================================
#  scripts/run_all_grid.sh — Validate the pipeline on a synthetic NxN grid.
#
#  Use this BEFORE scripts/run_all.sh. The grid is the cheap, deterministic
#  testbed: if PPO doesn't beat the baselines here, something is broken
#  upstream and there's no point burning hours training on the real net.
#
#  Tier isolation:
#      TSC_TIER=grid is exported so every step writes to
#        scenarios/grid/  outputs/grid/  models/grid/
#      Real-network results in outputs/real/ and models/real/ are NOT
#      touched by this script.
#
#  Usage:
#      ./scripts/run_all_grid.sh                       # offpeak, 4x4, seed=42
#      ./scripts/run_all_grid.sh rushhour              # different demand
#      GRID_SIZE=6 ./scripts/run_all_grid.sh           # 6x6 grid
#      TIMESTEPS=20000 ./scripts/run_all_grid.sh       # smoke test
#      SEED=7 TIMESTEPS=200000 ./scripts/run_all_grid.sh
#
#  Make executable once: chmod +x scripts/run_all_grid.sh
# =========================================================================

# Run from the repo root no matter where the script was invoked from.
cd "$(dirname "$0")/.." || exit 1

# --- Tier (do not change) -----------------------------------------------
export TSC_TIER=grid

# --- Configurable knobs (override via args or env) ----------------------
SCENARIO="${1:-offpeak}"
SEED="${SEED:-42}"
TIMESTEPS="${TIMESTEPS:-500000}"
GRID_SIZE="${GRID_SIZE:-4}"

STARTTIME=$(date +"%H:%M:%S")

echo "========================================================================="
echo " GRID validation pipeline   (tier=${TSC_TIER})"
echo " scenario=${SCENARIO}  size=${GRID_SIZE}x${GRID_SIZE}  seed=${SEED}  timesteps=${TIMESTEPS}"
echo "========================================================================="

# =========================================================================
echo
echo " STEP 1/8 : Generate ${GRID_SIZE}x${GRID_SIZE} grid network and scenarios"
echo "========================================================================="
python generate_grid_scenario.py \
    --size "${GRID_SIZE}" \
    --scenario "${SCENARIO}" \
    --tier "${TSC_TIER}" \
    --net-out networks/grid.net.xml \
    || { echo "[FAIL] grid generation failed. Aborting."; exit 1; }

echo
echo "========================================================================="
echo " STEP 2/8 : Fixed-time baseline (No Preempt)"
echo "========================================================================="
python run.py fixed       --scenario "${SCENARIO}"             || echo "[WARN] fixed exited with errors."
python run.py fixed       --scenario "${SCENARIO}" --emv       || echo "[WARN] fixed exited with errors."

echo
echo "========================================================================="
echo " STEP 3/8 : Fixed-time + EMV preemption"
echo "========================================================================="
python run.py fixed       --scenario "${SCENARIO}" --emv --preempt  || echo "[WARN] fixed --emv exited with errors."

echo
echo "========================================================================="
echo " STEP 4/8 : MaxPressure baseline (No Preempt)"
echo "========================================================================="
python run.py maxpressure --scenario "${SCENARIO}"             || echo "[WARN] maxpressure exited with errors."
python run.py maxpressure --scenario "${SCENARIO}"  --emv      || echo "[WARN] maxpressure exited with errors."

echo
echo "========================================================================="
echo " STEP 5/8 : MaxPressure + EMV preemption"
echo "========================================================================="
python run.py maxpressure --scenario "${SCENARIO}" --emv --preempt || echo "[WARN] maxpressure --emv exited with errors."

echo
echo "========================================================================="
echo " STEP 6/8 : Train PPO  (~minutes on the grid; longer for larger sizes)"
echo "========================================================================="
TRAIN_OK=1
python run.py train --scenario "${SCENARIO}" --emv --timesteps "${TIMESTEPS}" --seed "${SEED}" \
    || { echo "[FAIL] train failed. Skipping PPO eval and hybrid."; TRAIN_OK=0; }

if [ "${TRAIN_OK}" = "1" ]; then
    echo
    echo "========================================================================="
    echo " STEP 7a/8 : Evaluate PPO (pure learned policy)"
    echo "========================================================================="
    python run.py ppo    --scenario "${SCENARIO}" --emv  || echo "[WARN] ppo exited with errors."

    echo
    echo "========================================================================="
    echo " STEP 7b/8 : Hybrid PPO + EMV preemption"
    echo "========================================================================="
    python run.py hybrid --scenario "${SCENARIO}" --emv  || echo "[WARN] hybrid exited with errors."
fi

echo
echo "========================================================================="
echo " STEP 8/8 : Final Comparison"
echo "========================================================================="
python run.py evaluate --scenario "${SCENARIO}"  || echo "[WARN] evaluate exited with errors."

ENDTIME=$(date +"%H:%M:%S")

echo
echo "========================================================================="
echo " GRID PIPELINE COMPLETE   (tier=${TSC_TIER})"
echo " Scenario: ${SCENARIO}   Grid: ${GRID_SIZE}x${GRID_SIZE}"
echo " Seed: ${SEED}           Timesteps: ${TIMESTEPS}"
echo " Started : ${STARTTIME}"
echo " Finished: ${ENDTIME}"
echo
echo " Network : networks/grid.net.xml"
echo " Routes  : scenarios/${TSC_TIER}/${SCENARIO}.rou.xml , scenarios/${TSC_TIER}/emv.rou.xml"
echo
echo " Output files in: outputs/${TSC_TIER}/"
echo "   fixed_${SCENARIO}_tripinfo.xml               fixed_${SCENARIO}_emissions.xml"
echo "   fixed_emv_${SCENARIO}_emv_tripinfo.xml       fixed_emv_${SCENARIO}_emv_emissions.xml"
echo "   maxpressure_${SCENARIO}_tripinfo.xml         maxpressure_${SCENARIO}_emissions.xml"
echo "   maxpressure_emv_${SCENARIO}_emv_tripinfo.xml maxpressure_emv_${SCENARIO}_emv_emissions.xml"
echo "   ppo_eval_${SCENARIO}_emv_tripinfo.xml        ppo_eval_${SCENARIO}_emv_emissions.xml"
echo "   ppo_hybrid_${SCENARIO}_emv_tripinfo.xml      ppo_hybrid_${SCENARIO}_emv_emissions.xml"
echo
echo " Models  : models/${TSC_TIER}/"
echo
echo " Once these numbers look reasonable (PPO beats baselines, EMV waiting"
echo " drops to ~0s with preemption), switch to scripts/run_all.sh for the"
echo " production results."
echo "========================================================================="