"""
MaxPressure baseline for traffic signal control.

For each traffic light, at every control step, pick the green phase whose
movements maximize (upstream queue - downstream queue). Parameter-free;
theoretically stabilizing under moderate demand.

Strong baseline. If your RL doesn't beat this, your RL isn't ready.

Usage:
    python run_maxpressure.py                  # uses default Config()
    python run.py maxpressure --scenario offpeak          # via unified CLI
    python run.py maxpressure --scenario rushhour --emv   # different scenario
"""

import os
import sumo_setup  # noqa: F401
import traci
import sumolib
from sumolib import checkBinary

from config import Config
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary


class MaxPressureController:
    """One controller per traffic light.

    The `net` argument is a pre-loaded sumolib net (from
    `sumolib.net.readNet(..., withPrograms=True)`). It replaces the old
    module-level `_NET` global so that multiple runs with different net
    files can coexist without stale state.
    """

    def __init__(self, tls_id: str, min_green: int, yellow_time: int, net):
        self.tid = tls_id
        self.min_green = min_green
        self.yellow_time = yellow_time

        tls = net.getTLS(tls_id)
        programs = tls.getPrograms()
        program = next(iter(programs.values()))
        self.phases = program.getPhases()

        self.controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        self.green_phase_indices = [
            i for i, p in enumerate(self.phases)
            if ('G' in p.state or 'g' in p.state) and 'y' not in p.state
        ]

        self.current_phase = 0
        self.last_switch = 0
        self.in_yellow = False
        self.yellow_since = 0
        self.target_phase = 0

        # ✅ NEW: fairness tracking
        self.wait_time = {i: 0 for i in self.green_phase_indices}

        # ✅ NEW: tuning knobs
        self.fairness_weight = 0.5     # increase → more fairness
        self.max_wait = 60             # seconds before forced service
        self.spillback_threshold = 0.8 # downstream fullness ratio

    def resync(self, t: int):
        try:
            state = traci.trafficlight.getRedYellowGreenState(self.tid)
        except traci.TraCIException:
            return
        for idx, ph in enumerate(self.phases):
            if ph.state == state:
                self.current_phase = idx
                break
        self.in_yellow = False
        self.target_phase = self.current_phase
        self.last_switch = t

    def pressure(self, phase_index: int) -> float:
        state = self.phases[phase_index].state
        total = 0.0

        for link_idx, link_group in enumerate(self.controlled_links):
            if link_idx >= len(state) or state[link_idx] not in ('G', 'g'):
                continue

            for (in_lane, out_lane, _) in link_group:
                up = traci.lane.getLastStepHaltingNumber(in_lane)

                # ✅ anti-spillback
                if out_lane:
                    down = traci.lane.getLastStepHaltingNumber(out_lane)
                    length = traci.lane.getLength(out_lane)

                    # approximate capacity (7.5m per vehicle)
                    cap = max(length / 7.5, 1)

                    if down / cap > self.spillback_threshold:
                        continue  # skip blocked movement

                total += up  # NOTE: no longer subtracting downstream

        return total

    def step(self, t: int):
        if self.in_yellow:
            if t - self.yellow_since >= self.yellow_time:
                traci.trafficlight.setRedYellowGreenState(
                    self.tid, self.phases[self.target_phase].state
                )
                self.current_phase = self.target_phase
                self.in_yellow = False
                self.last_switch = t

                # ✅ reset wait time for chosen phase
                self.wait_time[self.current_phase] = 0
            return

        # update wait times
        for p in self.green_phase_indices:
            if p != self.current_phase:
                self.wait_time[p] += 1

        if t - self.last_switch < self.min_green:
            return

        best_phase = self.current_phase
        best_score = -float('inf')

        # ✅ check forced service (anti-starvation)
        for p in self.green_phase_indices:
            if self.wait_time[p] >= self.max_wait:
                best_phase = p
                break
        else:
            # normal selection with fairness
            for idx in self.green_phase_indices:
                p = self.pressure(idx)
                score = p + self.fairness_weight * self.wait_time[idx]

                if score > best_score:
                    best_score = score
                    best_phase = idx

        if best_phase != self.current_phase:
            cur_state = self.phases[self.current_phase].state
            yellow = ''.join('y' if c in ('G', 'g') else c for c in cur_state)

            traci.trafficlight.setRedYellowGreenState(self.tid, yellow)
            self.in_yellow = True
            self.yellow_since = t
            self.target_phase = best_phase

def run(cfg: Config, gui: bool = False):
    # Resolve everything from cfg up front so the rest of the function
    # doesn't care about module-level constants.
    net_file    = cfg.network.net_file
    route_files = cfg.route_files_str()
    num_seconds = cfg.sim.num_seconds
    teleport_s  = cfg.sim.time_to_teleport
    min_green   = cfg.fixed_time.min_green
    yellow_time = cfg.fixed_time.yellow_time
    out_prefix  = cfg.out_prefix("maxpressure")

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Load net ONCE for this run (was previously loaded at module import).
    # Doing it here makes multi-run scripts (swap networks between runs)
    # safe without stale globals.
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

    controllers = [
        MaxPressureController(tid, min_green, yellow_time, net)
        for tid in traci.trafficlight.getIDList()
    ]
    print(f'MaxPressure controlling {len(controllers)} traffic lights.')
    print(f'  Scenario: {cfg.scenario.name}  (EMV={cfg.scenario.emv})')

    for t in range(num_seconds):
        traci.simulationStep()
        for c in controllers:
            c.step(t)

    traci.close()
    print(f'Done. Outputs: {out_prefix}_tripinfo.xml, {out_prefix}_emissions.xml')
    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='Max_Pressure')


if __name__ == '__main__':
    # Runs with default config when invoked directly.
    # Use `python run.py maxpressure [...]` for CLI flag control.
    run(Config(), gui=False)