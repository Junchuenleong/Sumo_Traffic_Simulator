# Commands cheatsheet

Quick reference for running the pipeline. Pick a section, copy the block, run.

## `--emv` vs `--preempt` — read this first

These are **independent** flags:

| Flag         | What it does                                             |
|--------------|----------------------------------------------------------|
| `--emv`      | Adds `emv.rou.xml` to the route files — EMVs *exist* in the simulation |
| `--preempt`  | Controller actively preempts traffic lights when an EMV is near |

Three meaningful combinations per controller:

```bat
python run.py fixed --scenario offpeak                       :: no EMVs at all
python run.py fixed --scenario offpeak --emv                 :: EMVs on road, lights ignore them
python run.py fixed --scenario offpeak --emv --preempt       :: EMVs on road, lights preempt
```

Same for `maxpressure`. The `hybrid` controller always preempts (that's the whole point), so `--preempt` is implicit there. PPO eval (`ppo`) does not preempt — use `hybrid` if you want preemption layered on top of the trained policy.

`--preempt` without `--emv` is meaningless (no ambulances to react to) and prints a warning.

---

## Setting `TSC_TIER`

The env var that decides whether everything reads/writes under `grid/` or `real/`.
Default if unset: `real`. Both `run_all_*.bat` set it for you.

| shell      | grid                          | real                          |
|------------|-------------------------------|-------------------------------|
| cmd.exe    | `set TSC_TIER=grid`           | `set TSC_TIER=real`           |
| PowerShell | `$env:TSC_TIER="grid"`        | `$env:TSC_TIER="real"`        |
| bash/zsh   | `export TSC_TIER=grid`        | `export TSC_TIER=real`        |

One-shot (just for the next command, doesn't persist):

```bash
TSC_TIER=grid python run.py fixed --scenario offpeak    # bash
```

```powershell
$env:TSC_TIER="grid"; python run.py fixed --scenario offpeak    # PowerShell, persists in this terminal
```

Check what's currently set:

```bat
echo %TSC_TIER%        :: cmd
```
```powershell
echo $env:TSC_TIER     # PowerShell
```
```bash
echo $TSC_TIER         # bash
```

---

## Grid pipeline (synthetic, fast — start here)

### One-command
```bat
run_all_grid.bat                     :: defaults: offpeak, 4x4, seed=42, 100k steps
run_all_grid.bat rushhour            :: different demand
set GRID_SIZE=6 && run_all_grid.bat  :: 6x6 grid
set TIMESTEPS=20000 && run_all_grid.bat   :: smoke test
```

### Individual commands
```bat
set TSC_TIER=grid

python generate_grid_scenario.py --scenario offpeak
python generate_grid_scenario.py --size 6 --scenario rushhour       :: bigger grid

:: Baselines, no EMVs
python run.py fixed       --scenario offpeak
python run.py maxpressure --scenario offpeak

:: Baselines, EMVs on road, controller IGNORES them
python run.py fixed       --scenario offpeak --emv
python run.py maxpressure --scenario offpeak --emv

:: Baselines + preemption
python run.py fixed       --scenario offpeak --emv --preempt
python run.py maxpressure --scenario offpeak --emv --preempt

:: PPO
python run.py train       --scenario offpeak --emv --timesteps 100000 --seed 42
python run.py ppo         --scenario offpeak --emv
python run.py hybrid      --scenario offpeak --emv

python run.py evaluate    --scenario offpeak
```

---

## Real-network pipeline (Semenyih or your own OSM city)

### One-command
```bat
run_all.bat                              :: defaults: offpeak, seed=42, 500k steps
run_all.bat rushhour                     :: different scenario
set TIMESTEPS=10000 && run_all.bat       :: smoke test
set SEED=7 && set TIMESTEPS=1000000 && run_all.bat   :: longer run, different seed
```

### Individual commands
```bat
set TSC_TIER=real

python run.py generate                          :: routes only (safe to re-run)
python run.py generate --regen-tls              :: ⚠ rebuilds TLS programs (destructive)

:: Baselines, no EMVs
python run.py fixed       --scenario offpeak
python run.py maxpressure --scenario offpeak

:: Baselines, EMVs on road, controller IGNORES them
python run.py fixed       --scenario offpeak --emv
python run.py maxpressure --scenario offpeak --emv

:: Baselines + preemption
python run.py fixed       --scenario offpeak --emv --preempt
python run.py maxpressure --scenario offpeak --emv --preempt

:: PPO
python run.py train       --scenario offpeak --emv --timesteps 500000 --seed 42
python run.py ppo         --scenario offpeak --emv
python run.py hybrid      --scenario offpeak --emv

python run.py evaluate    --scenario offpeak
```

---

## Switching tiers mid-session

Just flip the env var — no `config.py` edits, no script changes:

```bat
set TSC_TIER=grid          :: now reading/writing grid/
python run.py fixed --scenario offpeak

set TSC_TIER=real          :: now reading/writing real/
python run.py fixed --scenario offpeak
```

---

## Watching runs in SUMO-GUI

Append `--gui` to any controller command. The GUI runs the same simulation but
with a window so you can eyeball traffic, watch preemption fire, or step
through frame-by-frame. Expect a ~5-10× slowdown — useful for inspection,
not for collecting metrics.

```bat
python run.py fixed       --scenario offpeak --gui
python run.py fixed       --scenario offpeak --emv --gui                :: EMVs visible, no preempt
python run.py fixed       --scenario offpeak --emv --preempt --gui      :: watch preemption fire
python run.py maxpressure --scenario rushhour --gui
python run.py maxpressure --scenario offpeak --emv --preempt --gui
python run.py ppo         --scenario offpeak --emv --gui                :: watch trained policy
python run.py hybrid      --scenario offpeak --emv --gui                :: PPO + preemption
```

GUI tips:
- Hit **Run** (▶) in the SUMO-GUI window to start stepping. Use the speed
  slider to slow it down — preemption flips happen in 1-2 seconds.
- **View → Show controllable lanes** colours TLS-controlled lanes so you can
  see which ones the agent is actually steering.
- Right-click an EMV (red vehicle) → **Track** to follow it across the network.
- Closing the GUI window mid-run kills the simulation cleanly.

`--gui` works on `fixed`, `maxpressure`, `ppo`, and `hybrid`. It does **not**
work on `train` (you don't want a window for 500k steps) or `evaluate`
(no simulation, just XML parsing).

---

## Diagnostics & sanity checks

```bat
:: TLS phase counts — confirms padded action space looks right (run once per net)
python train_ppo.py --sanity

:: EMV route diagnostic — which TLSs each ambulance crosses
python run.py hybrid --scenario offpeak --emv --inspect-emv

:: Verbose preemption logging (firing, releasing, ETA values)
python run.py hybrid --scenario offpeak --emv --debug-preempt

:: Both at once — and watch it happen
python run.py hybrid --scenario offpeak --emv --inspect-emv --debug-preempt --gui
```

---

## Tweaking preemption knobs without editing config.py

```bat
python run.py hybrid --scenario offpeak --emv --detection-distance 300
python run.py hybrid --scenario offpeak --emv --eta-threshold 20
python run.py hybrid --scenario offpeak --emv --detection-distance 300 --eta-threshold 20
```

Same flags work on `fixed --emv --preempt` and `maxpressure --emv --preempt`.

---

## TensorBoard

```bat
tensorboard --logdir outputs/grid/tb/      :: grid runs
tensorboard --logdir outputs/real/tb/      :: real runs
```

---

## VS Code Run button

Add to `.vscode/launch.json` so the green play button respects tier:

```json
{
  "version": "0.2.0",
  "configurations": [
    { "name": "Python: Current File (grid)",
      "type": "debugpy", "request": "launch",
      "program": "${file}", "console": "integratedTerminal",
      "env": { "TSC_TIER": "grid" } },
    { "name": "Python: Current File (real)",
      "type": "debugpy", "request": "launch",
      "program": "${file}", "console": "integratedTerminal",
      "env": { "TSC_TIER": "real" } }
  ]
}
```

Pick `(grid)` or `(real)` in the Run & Debug dropdown before hitting play.

---

## Common workflows

**Just cloned the repo — does it work?**
```bat
set TSC_TIER=grid
python generate_grid_scenario.py --scenario offpeak
python run.py fixed --scenario offpeak
```
Should finish in seconds and write `outputs/grid/fixed_offpeak_tripinfo.xml`.

**Quick smoke test after a code change** (grid, 20k steps):
```bat
set TIMESTEPS=20000 && run_all_grid.bat
```

**Reproduce paper numbers** (real, 500k steps):
```bat
run_all.bat
```

**Just compare existing results** (no re-running):
```bat
set TSC_TIER=real
python run.py evaluate --scenario offpeak
```

**Watch the trained policy in action** (real, with preemption, GUI):
```bat
set TSC_TIER=real
python run.py hybrid --scenario offpeak --emv --gui
```

**Compare EMV travel time across all controllers** (the canonical EMV experiment):
```bat
:: All four runs include EMVs; only the --preempt rows let lights react.
python run.py fixed       --scenario offpeak --emv
python run.py fixed       --scenario offpeak --emv --preempt
python run.py maxpressure --scenario offpeak --emv
python run.py maxpressure --scenario offpeak --emv --preempt
python run.py hybrid      --scenario offpeak --emv
python run.py evaluate    --scenario offpeak
```

**Fresh OSM city — first-time setup**
```bat
:: Edit _TIER_NETS["real"] in config.py to point at your new .net.xml
set TSC_TIER=real
python run.py generate
python run.py fixed --scenario offpeak       :: sanity check
:: then full pipeline:
run_all.bat
```

---

## Output locations (so you know where to look)

| What                          | Path                                          |
|-------------------------------|-----------------------------------------------|
| Tripinfo XMLs                 | `outputs/<tier>/<controller>_<scenario>_tripinfo.xml`     |
| Emissions XMLs                | `outputs/<tier>/<controller>_<scenario>_emissions.xml`    |
| TensorBoard logs              | `outputs/<tier>/tb/`                          |
| Final PPO model               | `models/<tier>/ppo_shared_final.zip`          |
| Best PPO model (eval-picked)  | `models/<tier>/best/best_model.zip`           |
| VecNormalize stats            | `models/<tier>/vecnormalize.pkl`              |
| Phase log (hybrid)            | `outputs/<tier>/phase_log_ppo_<scenario>*.csv` |
| Generated grid net            | `networks/grid.net.xml`                       |
| Generated route files         | `scenarios/<tier>/<scenario>.rou.xml`         |
| EMV route file                | `scenarios/<tier>/emv.rou.xml`                |

When `--preempt` is used, the controller name in the path becomes `<controller>_emv`, e.g. `outputs/real/fixed_emv_offpeak_emv_tripinfo.xml`. So the no-preempt and preempt runs never collide on disk.