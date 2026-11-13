@echo off
REM =========================================================================
REM  run_all_grid.bat — Validate the pipeline on a synthetic NxN grid.
REM
REM  Use this BEFORE run_all.bat. The grid is the cheap, deterministic
REM  testbed: if PPO doesn't beat the baselines here, something is broken
REM  upstream and there's no point burning hours training on the real net.
REM
REM  Tier isolation:
REM      TSC_TIER=grid is set at the top so every step writes to
REM        scenarios/grid/  outputs/grid/  models/grid/
REM      Real-network results in outputs/real/ and models/real/ are NOT
REM      touched by this script.
REM
REM  Prerequisite (one-time):
REM      Edit config.py so NetworkConfig.net_file points at the grid net:
REM          net_file: str = str(NETWORKS_DIR / "grid.net.xml")
REM
REM  Usage:
REM      run_all_grid.bat                       REM defaults: offpeak, 4x4, seed=42
REM      run_all_grid.bat rushhour              REM scenario=rushhour
REM      set GRID_SIZE=6 ^& run_all_grid.bat    REM 6x6 grid
REM      set TIMESTEPS=20000 ^& run_all_grid.bat REM smoke test
REM =========================================================================

setlocal
set STARTTIME=%TIME%

REM --- Tier (do not change) ------------------------------------------------
set TSC_TIER=grid

REM --- Configurable knobs (override via args or env) -----------------------
set SCENARIO=%1
if "%SCENARIO%"=="" set SCENARIO=offpeak
if "%SEED%"=="" set SEED=42
if "%TIMESTEPS%"=="" set TIMESTEPS=500000
if "%GRID_SIZE%"=="" set GRID_SIZE=4

echo =========================================================================
echo  GRID validation pipeline   (tier=%TSC_TIER%)
echo  scenario=%SCENARIO%  size=%GRID_SIZE%x%GRID_SIZE%  seed=%SEED%  timesteps=%TIMESTEPS%
echo =========================================================================
echo  Reminder: NetworkConfig.net_file in config.py must point at
echo            networks/grid.net.xml for this pipeline to make sense.
echo =========================================================================

REM =========================================================================
echo.
echo  STEP 1/8 : Generate %GRID_SIZE%x%GRID_SIZE% grid network and scenarios
echo =========================================================================
python generate_grid_scenario.py --size %GRID_SIZE% --scenario %SCENARIO% --tier %TSC_TIER% --net-out networks/grid.net.xml
if %ERRORLEVEL% neq 0 (
    echo [FAIL] grid generation failed. Aborting.
    goto :DONE
)

echo.
echo =========================================================================
echo  STEP 2/8 : Fixed-time baseline (with EMV no preempt)
echo =========================================================================
python run.py fixed       --scenario %SCENARIO%
python run.py fixed       --scenario %SCENARIO% --emv
if %ERRORLEVEL% neq 0 echo [WARN] fixed exited with errors.

echo.
echo =========================================================================
echo  STEP 3/8 : Fixed-time + EMV preemption
echo =========================================================================
python run.py fixed       --scenario %SCENARIO% --emv --preempt
if %ERRORLEVEL% neq 0 echo [WARN] fixed --emv exited with errors.

echo.
echo =========================================================================
echo  STEP 4/8 : MaxPressure baseline (with EMV no preempt)
echo =========================================================================
python run.py maxpressure --scenario %SCENARIO%
python run.py maxpressure --scenario %SCENARIO% --emv
if %ERRORLEVEL% neq 0 echo [WARN] maxpressure exited with errors.

echo.
echo =========================================================================
echo  STEP 5/8 : MaxPressure + EMV preemption
echo =========================================================================
python run.py maxpressure --scenario %SCENARIO% --emv --preempt
if %ERRORLEVEL% neq 0 echo [WARN] maxpressure --emv exited with errors.

echo.
echo =========================================================================
echo  STEP 6/8 : Train PPO  (~minutes on the grid; longer for larger sizes)
echo =========================================================================
python run.py train       --scenario %SCENARIO% --emv --timesteps %TIMESTEPS% --seed %SEED%
if %ERRORLEVEL% neq 0 (
    echo [FAIL] train failed. Skipping PPO eval and hybrid.
    goto :EVALUATE
)

echo.
echo =========================================================================
echo  STEP 7a/8 : Evaluate PPO (pure learned policy)
echo =========================================================================
python run.py ppo         --scenario %SCENARIO% --emv
if %ERRORLEVEL% neq 0 echo [WARN] ppo exited with errors.

echo.
echo =========================================================================
echo  STEP 7b/8 : Hybrid PPO + EMV preemption
echo =========================================================================
python run.py hybrid      --scenario %SCENARIO% --emv
if %ERRORLEVEL% neq 0 echo [WARN] hybrid exited with errors.

:EVALUATE
echo.
echo =========================================================================
echo  STEP 8/8 : Final Comparison
echo =========================================================================
python run.py evaluate    --scenario %SCENARIO%
if %ERRORLEVEL% neq 0 echo [WARN] evaluate exited with errors.

:DONE
echo.
echo =========================================================================
echo  GRID PIPELINE COMPLETE   (tier=%TSC_TIER%)
echo  Scenario: %SCENARIO%   Grid: %GRID_SIZE%x%GRID_SIZE%
echo  Seed: %SEED%           Timesteps: %TIMESTEPS%
echo  Started : %STARTTIME%
echo  Finished: %TIME%
echo.
echo  Network : networks/grid.net.xml
echo  Routes  : scenarios/%TSC_TIER%/%SCENARIO%.rou.xml , scenarios/%TSC_TIER%/emv.rou.xml
echo.
echo  Output files in: outputs/%TSC_TIER%/
echo    fixed_%SCENARIO%_tripinfo.xml               fixed_%SCENARIO%_emissions.xml
echo    fixed_emv_%SCENARIO%_emv_tripinfo.xml       fixed_emv_%SCENARIO%_emv_emissions.xml
echo    maxpressure_%SCENARIO%_tripinfo.xml         maxpressure_%SCENARIO%_emissions.xml
echo    maxpressure_emv_%SCENARIO%_emv_tripinfo.xml maxpressure_emv_%SCENARIO%_emv_emissions.xml
echo    ppo_eval_%SCENARIO%_emv_tripinfo.xml        ppo_eval_%SCENARIO%_emv_emissions.xml
echo    ppo_hybrid_%SCENARIO%_emv_tripinfo.xml      ppo_hybrid_%SCENARIO%_emv_emissions.xml
echo.
echo  Models  : models/%TSC_TIER%/
echo.
echo  Once these numbers look reasonable (PPO beats baselines, EMV waiting
echo  drops to ~0s with preemption), run run_all.bat for the real world production results.
echo =========================================================================

endlocal
pause
