"""
Fixed-time baseline. Just runs SUMO with whatever cyclic phase program
netconvert inferred. No control code — the dumbest baseline.

Usage:
    python run_fixed.py                       # uses default Config()
    python run.py fixed --scenario offpeak    # via unified CLI
    python run.py fixed --scenario rushhour
"""

import os
import sumo_setup  # noqa: F401
import traci
from sumolib import checkBinary

from config import Config
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary

def run(cfg: Config, gui: bool = False):
    net_file    = cfg.network.net_file
    route_files = cfg.route_files_str()
    num_seconds = cfg.sim.num_seconds
    teleport_s  = cfg.sim.time_to_teleport
    out_prefix  = cfg.out_prefix("fixed")

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    sumo_bin = checkBinary('sumo-gui' if gui else 'sumo')
    traci.start([
        sumo_bin, '-n', net_file, '-r', route_files,
        '--no-step-log', 'true',
        '--waiting-time-memory', '10000',
        '--tripinfo-output', f'{out_prefix}_tripinfo.xml',
        '--emission-output', f'{out_prefix}_emissions.xml',
        '--time-to-teleport', str(teleport_s),
    ])
    print(f'Fixed-time baseline.')
    print(f'  Scenario: {cfg.scenario.name}  (EMV={cfg.scenario.emv})')

    for _ in range(num_seconds):
        traci.simulationStep()

    traci.close()
    print(f'Done. Outputs: {out_prefix}_tripinfo.xml, {out_prefix}_emissions.xml')
    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='fixed')


if __name__ == '__main__':
    run(Config(), gui=False)