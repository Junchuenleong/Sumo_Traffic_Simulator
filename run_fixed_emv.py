"""
Fixed-time + Emergency Vehicle Preemption.

Refactored to accept a Config object so `run.py` can drive it with any
scenario / preemption settings from the CLI. Still runnable standalone:
`python run_fixed_emv.py` uses the default Config().
"""

import os
import sumo_setup  # noqa: F401
import traci
from sumolib import checkBinary

from config import Config, EmergencyPreemption
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary


def run(cfg: Config, gui: bool = False):
    # Resolve everything from cfg up front so the rest of the function
    # doesn't care about module-level constants.
    net_file      = cfg.network.net_file
    route_files   = cfg.route_files_str()
    num_seconds   = cfg.sim.num_seconds
    teleport_s    = cfg.sim.time_to_teleport
    out_prefix    = cfg.out_prefix("fixed_emv")

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

    tls_ids = traci.trafficlight.getIDList()
    preemption = {
        tid: EmergencyPreemption(
            tid,
            detection_distance=cfg.preemption.detection_distance,
            release_distance=cfg.preemption.release_distance,
            eta_threshold_s=cfg.preemption.eta_threshold_s,
            always_preempt_within_m=cfg.preemption.always_preempt_within_m,
            detection_probability=cfg.preemption.detection_probability,
            comm_latency_steps=cfg.preemption.comm_latency_steps,
            debug=cfg.preemption.debug,
        )
        for tid in tls_ids
    }
    print(f'Fixed-time + EMV preemption across {len(tls_ids)} traffic lights.')
    print(f'  Scenario: {cfg.scenario.name}  (EMV={cfg.scenario.emv})')

    for _ in range(num_seconds):
        traci.simulationStep()
        for tid in tls_ids:
            preemption[tid].step()

    traci.close()
    print(f'Outputs: {out_prefix}_tripinfo.xml, {out_prefix}_emissions.xml')
    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='fixed_preempt')


if __name__ == '__main__':
    # Run with default config when invoked directly.
    # Use run.py for CLI flag control.
    run(Config(scenario=__import__('config').ScenarioConfig(name="offpeak", emv=True)),
        gui=False)