"""
Parse SUMO tripinfo / emission XMLs and print a comparison table.

Metrics reported:
  - Number of trips completed (throughput proxy)
  - Average waiting time per vehicle (seconds stopped)
  - Average trip duration (seconds)
  - Average time loss (seconds lost vs. free-flow ideal)
  - Total CO2 emitted (kg)

Setting `TSC_TIER`

The env var that decides whether everything reads/writes under `grid/` or `real/`.
Default if unset: `real`. Both `run_all_*.bat` set it for you.

| shell      | grid                          | real                          |
|------------|-------------------------------|-------------------------------|
| cmd.exe    | set TSC_TIER=grid           | set TSC_TIER=real           |
| PowerShell | $env:TSC_TIER="grid"        | $env:TSC_TIER="real"        |
| bash/zsh   | export TSC_TIER=grid        | export TSC_TIER=real        |

Usage:
    python evaluate.py                                 # default: offpeak
    python run.py evaluate                             # same
    python run.py evaluate --scenario rushhour
    python run.py evaluate --scenario offpeak

Assumes you've already run the controllers whose results you want compared,
e.g.:
    python run.py fixed       --scenario offpeak
    python run.py fixed       --scenario offpeak --emv
    python run.py maxpressure --scenario offpeak
    python run.py maxpressure --scenario offpeak --emv
    python run.py ppo         --scenario offpeak --emv
    python run.py hybrid      --scenario offpeak --emv
"""

import argparse
import os
import xml.etree.ElementTree as ET

from config import Config, ScenarioConfig


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_tripinfo(path: str):
    if not os.path.exists(path):
        return None
    trips = ET.parse(path).getroot().findall('tripinfo')
    if not trips:
        return {'n_trips': 0, 'avg_wait_s': 0, 'avg_duration_s': 0, 'avg_timeloss_s': 0}
    waits = [float(t.get('waitingTime', 0)) for t in trips]
    durs  = [float(t.get('duration',    0)) for t in trips]
    loss  = [float(t.get('timeLoss',    0)) for t in trips]
    n = len(trips)
    return {
        'n_trips':        n,
        'avg_wait_s':     sum(waits) / n,
        'avg_duration_s': sum(durs)  / n,
        'avg_timeloss_s': sum(loss)  / n,
    }


def parse_total_co2_kg(path: str):
    """Stream-parse the emissions file (can be huge) and sum CO2."""
    if not os.path.exists(path):
        return None
    total_mg = 0.0
    for _, elem in ET.iterparse(path, events=('end',)):
        if elem.tag == 'vehicle':
            total_mg += float(elem.get('CO2', 0))
            elem.clear()
    return total_mg / 1e6  # mg -> kg


# ---------------------------------------------------------------------------
# File discovery — uses Config.out_prefix() so this code and the run scripts
# agree on naming by construction. If the run scripts' output paths change,
# this file follows automatically.
# ---------------------------------------------------------------------------

def find_existing_prefix(candidates):
    """Return the first prefix whose _tripinfo.xml exists, or None."""
    for p in candidates:
        if os.path.exists(f'{p}_tripinfo.xml'):
            return p
    return None


def build_runs(scenario: str):
    """Resolve (display_name, file_prefix) pairs for the given scenario.

    Three groups of rows:

      1. Clean baselines (no EMVs in routes)        — overall network performance
      2. Baselines with EMVs but NO preemption      — EMV travel time without intervention
      3. Treatments with EMVs AND preemption        — EMV travel time with intervention

    Group 1 uses cfg_no (emv=False); groups 2 and 3 use cfg_emv (emv=True).
    Group 2 vs group 3 is distinguished by which controller wrote the file:
    plain `fixed`/`maxpressure` (no preempt) vs `fixed_emv`/`maxpressure_emv` (preempt).
    """
    cfg_no  = Config(scenario=ScenarioConfig(name=scenario, emv=False))
    cfg_emv = Config(scenario=ScenarioConfig(name=scenario, emv=True))

    return [
        # 1. Clean baselines — no EMVs at all
        ('Fixed-time',            [cfg_no.out_prefix('fixed')]),
        ('MaxPressure',           [cfg_no.out_prefix('maxpressure')]),

        # 2. Baselines with EMVs on road, controller IGNORES them
        ('Fixed-time + EMV',      [cfg_emv.out_prefix('fixed')]),
        ('MaxPressure + EMV',     [cfg_emv.out_prefix('maxpressure')]),
        ('PPO',                   [cfg_emv.out_prefix('ppo_eval')]),

        # 3. Treatments with EMVs on road, controller PREEMPTS
        ('Fixed-time + preempt',  [cfg_emv.out_prefix('fixed_emv')]),
        ('MaxPressure + preempt', [cfg_emv.out_prefix('maxpressure_emv')]),
        ('Hybrid PPO',            [cfg_emv.out_prefix('ppo_hybrid')]),
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(scenario: str = 'offpeak'):
    Config(scenario=ScenarioConfig(name=scenario)).banner_tier()
    print(f'\nComparison for scenario: {scenario}, please ensure you are on the correct tier.\n')

    rows = []
    for name, candidates in build_runs(scenario):
        prefix = find_existing_prefix(candidates)
        if prefix is None:
            tried = ', '.join(f'{c}_tripinfo.xml' for c in candidates)
            print(f'[skip] {name}: not found (looked for {tried})')
            continue
        t = parse_tripinfo(f'{prefix}_tripinfo.xml')
        co2 = parse_total_co2_kg(f'{prefix}_emissions.xml')
        rows.append((name, t, co2))

    if not rows:
        print(f'\nNo runs found for scenario "{scenario}". Example commands:')
        print(f'  python run.py fixed       --scenario {scenario}')
        print(f'  python run.py maxpressure --scenario {scenario}')
        print(f'  python run.py fixed       --scenario {scenario} --emv')
        print(f'  python run.py maxpressure --scenario {scenario} --emv')
        print(f'  python run.py fixed       --scenario {scenario} --emv --preempt')
        print(f'  python run.py maxpressure --scenario {scenario} --emv --preempt')
        print(f'  python run.py ppo         --scenario {scenario} --emv')
        print(f'  python run.py hybrid      --scenario {scenario} --emv')
        return

    print()
    print(f"{'Method':<20} {'Trips':>6} {'AvgWait(s)':>11} {'AvgDur(s)':>10} "
          f"{'TimeLoss(s)':>12} {'CO2(kg)':>9}")
    print('-' * 74)
    for name, t, co2 in rows:
        co2_str = f'{co2:>9.1f}' if co2 is not None else '        -'
        print(f"{name:<20} {t['n_trips']:>6d} {t['avg_wait_s']:>11.2f} "
              f"{t['avg_duration_s']:>10.2f} {t['avg_timeloss_s']:>12.2f} {co2_str}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare controller results (tripinfo + emissions).'
    )
    parser.add_argument('--scenario', default='offpeak',
                        choices=['offpeak', 'rushhour', 'gridlock'],
                        help='Which scenario to compare (default: offpeak)')
    args = parser.parse_args()
    main(scenario=args.scenario)