@echo off
REM =========================================================================
REM  run_all.bat — Execute the full traffic signal control pipeline
REM
REM  Usage:
REM      run_all.bat                   REM defaults: scenario=offpeak, seed=42
REM      run_all.bat rushhour          REM scenario=rushhour
REM      set SEED=7 ^& run_all.bat     REM override seed via env var
REM
REM  Order:
REM    1. (optional) Generate scenario route files
REM    2. Run baselines (Fixed, MaxPressure) with and without EMV
REM    3. Train PPO
REM    4. Evaluate PPO (pure learned policy)
REM    5. Run hybrid PPO + EMV preemption
REM    6. Comparison table
REM =========================================================================

setlocal
set STARTTIME=%TIME%

REM --- Tier (do not change) ------------------------------------------------
set TSC_TIER=real

REM --- Configurable knobs (override via args or env) -----------------------
set SCENARIO=%1
if "%SCENARIO%"=="" set SCENARIO=offpeak
if "%SEED%"=="" set SEED=42
if "%TIMESTEPS%"=="" set TIMESTEPS=500000

echo =========================================================================
echo  Pipeline config: scenario=%SCENARIO%  seed=%SEED%  timesteps=%TIMESTEPS%
echo =========================================================================

REM =========================================================================
echo.
echo  STEP 1/8 : Generate network and traffic scenarios  (commented out)
echo =========================================================================
REM Uncomment the next line to (re)generate scenario .rou.xml files.
REM This is only needed the first time, or after editing generate_scenarios.py.
REM python run.py generate

echo.
echo =========================================================================
echo  STEP 2/8 : Fixed-time baselines (no Preempt)
echo =========================================================================
python run.py fixed       --scenario offpeak
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
echo  STEP 4/8 : MaxPressure baseline (no Preempt)
echo =========================================================================
python run.py maxpressure --scenario offpeak
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
echo  STEP 6/8 : Train PPO (this will take a while)
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
echo  ALL STEPS COMPLETE
echo  Scenario: %SCENARIO%   Seed: %SEED%   Timesteps: %TIMESTEPS%
echo  Started : %STARTTIME%
echo  Finished: %TIME%
echo.
echo  Output files in: outputs/
echo    fixed_%SCENARIO%_tripinfo.xml               fixed_%SCENARIO%_emissions.xml
echo    fixed_emv_%SCENARIO%_emv_tripinfo.xml       fixed_emv_%SCENARIO%_emv_emissions.xml
echo    maxpressure_%SCENARIO%_tripinfo.xml         maxpressure_%SCENARIO%_emissions.xml
echo    maxpressure_emv_%SCENARIO%_emv_tripinfo.xml maxpressure_emv_%SCENARIO%_emv_emissions.xml
echo    ppo_eval_%SCENARIO%_emv_tripinfo.xml        ppo_eval_%SCENARIO%_emv_emissions.xml
echo    ppo_hybrid_%SCENARIO%_emv_tripinfo.xml      ppo_hybrid_%SCENARIO%_emv_emissions.xml
echo =========================================================================

endlocal
pause