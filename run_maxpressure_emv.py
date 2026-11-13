"""
MaxPressure + Emergency Vehicle Preemption.
"""

import os
import sumo_setup  # noqa: F401
import traci
import sumolib
from sumolib import checkBinary

from run_maxpressure import MaxPressureController
from config import Config, EmergencyPreemption
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary


def run(cfg: Config, gui: bool = False):
    net_file    = cfg.network.net_file
    route_files = cfg.route_files_str()
    num_seconds = cfg.sim.num_seconds
    teleport_s  = cfg.sim.time_to_teleport
    min_green   = cfg.fixed_time.min_green
    yellow_time = cfg.fixed_time.yellow_time
    out_prefix  = cfg.out_prefix("maxpressure_emv")

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Load the static net once for this run.
    net = sumolib.net.readNet(net_file, withPrograms=True)

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
    controllers = {
        tid: MaxPressureController(tid, min_green, yellow_time, net)
        for tid in tls_ids
    }
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
    print(f'MaxPressure + EMV preemption across {len(tls_ids)} traffic lights.')
    print(f'  Scenario: {cfg.scenario.name}  (EMV={cfg.scenario.emv})')

    preempt_steps = 0
    for t in range(num_seconds):
        traci.simulationStep()
        for tid in tls_ids:
            pre = preemption[tid]
            was_active = pre.active
            if pre.step():
                preempt_steps += 1
            else:
                if was_active:                   # <-- release edge: resync
                    controllers[tid].resync(t)
                controllers[tid].step(t)

    traci.close()
    print(f'Done. Preemption was active for {preempt_steps} TLS-steps.')
    print(f'Outputs: {out_prefix}_tripinfo.xml, {out_prefix}_emissions.xml')
    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='Max_Pressure_Preempt')


if __name__ == '__main__':
    # Default standalone invocation runs offpeak + EMV (since this is the
    # EMV variant). Use `python run.py maxpressure --scenario ... --emv`
    # for other scenarios.
    from config import ScenarioConfig
    run(Config(scenario=ScenarioConfig(name="offpeak", emv=True)), gui=False)