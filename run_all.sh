#!/usr/bin/env bash
# =========================================================================
#  scripts/run_all.sh — Execute the full traffic signal control pipeline
#                      on the real (OSM-imported) network.
#
#  Tier isolation:
#      TSC_TIER=real is exported so every step writes to
#        scenarios/real/  outputs/real/  models/real/
#      Grid results in outputs/grid/ and models/grid/ are NOT touched.
#
#  Usage:
#      ./scripts/run_all.sh                       # offpeak, seed=42, 500k steps
#      ./scripts/run_all.sh rushhour              # different scenario
#      TIMESTEPS=10000 ./scripts/run_all.sh       # quick smoke test
#      SEED=7 TIMESTEPS=1000000 ./scripts/run_all.sh
#
#  First run on a new network: uncomment the `python run.py generate` line
#  in STEP 1 below. Once scenarios are built, leave it commented so you
#  don't overwrite them on every re-run.
#
#  Make executable once: chmod +x scripts/run_all.sh
# =========================================================================

# Run from the repo root no matter where the script was invoked from.
cd "$(dirname "$0")/.." || exit 1

# --- Tier (do not change) -----------------------------------------------
export TSC_TIER=real

# --- Configurable knobs (override via args or env) ----------------------
SCENARIO="${1:-offpeak}"
SEED="${SEED:-42}"
TIMESTEPS="${TIMESTEPS:-500000}"

STARTTIME=$(date +"%H:%M:%S")

echo "========================================================================="
echo " REAL-NETWORK pipeline   (tier=${TSC_TIER})"
echo " scenario=${SCENARIO}  seed=${SEED}  timesteps=${TIMESTEPS}"
echo "========================================================================="

# =========================================================================
echo
echo " STEP 1/8 : Generate scenarios  (commented out — uncomment for first run)"
echo "========================================================================="
# Uncomment the next line to (re)generate scenario .rou.xml files.
# This is only needed the first time, or after editing generate_scenarios.py.
# python run.py generate || { echo "[FAIL] generate failed. Aborting."; exit 1; }

echo
echo "========================================================================="
echo " STEP 2/8 : Fixed-time baseline (no Preempt)"
echo "========================================================================="
python run.py fixed       --scenario "${SCENARIO}"             || echo "[WARN] fixed exited with errors."
python run.py fixed       --scenario "${SCENARIO}" --emv       || echo "[WARN] fixed exited with errors."

echo
echo "========================================================================="
echo " STEP 3/8 : Fixed-time + EMV preemption"
echo "========================================================================="
python run.py fixed       --scenario "${SCENARIO}" --emv --preempt || echo "[WARN] fixed --emv exited with errors."

echo
echo "========================================================================="
echo " STEP 4/8 : MaxPressure baseline (no Preempt)"
echo "========================================================================="
python run.py maxpressure --scenario "${SCENARIO}"             || echo "[WARN] maxpressure exited with errors."
python run.py maxpressure --scenario "${SCENARIO}" --emv       || echo "[WARN] maxpressure exited with errors."

echo
echo "========================================================================="
echo " STEP 5/8 : MaxPressure + EMV preemption"
echo "========================================================================="
python run.py maxpressure --scenario "${SCENARIO}" --emv --preempt || echo "[WARN] maxpressure --emv exited with errors."

echo
echo "========================================================================="
echo " STEP 6/8 : Train PPO (this will take a while)"
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
echo " REAL-NETWORK PIPELINE COMPLETE   (tier=${TSC_TIER})"
echo " Scenario: ${SCENARIO}   Seed: ${SEED}   Timesteps: ${TIMESTEPS}"
echo " Started : ${STARTTIME}"
echo " Finished: ${ENDTIME}"
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
echo "========================================================================="