"""
Emergency Vehicle Preemption for traffic lights (v2).

Detection is VEHICLE-based, not lane-based. For every emergency-class
vehicle in the simulation we call traci.vehicle.getNextTLS(), which tells
us which traffic lights the vehicle will cross next on its current route
and at what route-following distance. We filter to our TLS, pick the
closest approaching EMV, and return the action index that gives that
EMV's link a green signal.

Why vehicle-centric is better than lane-centric:
  * Route-aware. A vehicle whose route does not include this TLS will
    not appear in our results, even if it is physically nearby. No false
    positives from EMVs that turn onto a different road before reaching
    us.
  * Correct distance. getNextTLS returns meters along the route to the
    stop line, not meters of the current lane. This works on networks
    where TLS-controlled lanes are short stubs (common for OSM imports
    after --tls.join).
  * The link index comes straight from SUMO, so we know exactly which
    position in the state string needs to be green.

Features in this module:
  * Hysteresis (detection_distance vs release_distance) to prevent
    flapping at the boundary.
  * Extension. If the current phase already serves the approaching EMV,
    the current action index is returned so sumo-rl holds it (no wasteful
    switch through yellow).
  * Optional real-world imperfections: detection dropout and comm
    latency. Defaults to ideal behaviour; turn them up to simulate real
    preemption hardware (GPS/AVL, optical, radio).
  * Multi-EMV arbitration. The closest approaching EMV wins; if two
    EMVs happen to need the same green, both are served automatically.
  * Rate-limited debug logging.

Public API: EmergencyPreemption(...).get_preempt_action(traffic_signal)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import traci


@dataclass
class EMVApproach:
    """A single emergency vehicle approaching this TLS."""
    vehicle_id: str
    link_idx: int      # position in the TLS state string
    distance: float    # route-following meters to stop line
    speed: float       # current m/s


class EmergencyPreemption:
    """Per-TLS preemption controller. Returns a sumo-rl action index (or
    None) each tick, to be used by the hybrid runner as an override."""

    def __init__(
        self,
        tls_id: str,
        detection_distance: float = 250.0,
        release_distance: float | None = None,
        # Realism knobs. Defaults = ideal; tune up to model real hardware.
        detection_probability: float = 1.0,
        comm_latency_steps: int = 0,
        rng_seed: int | None = None,
        debug: bool = False,
        yellow_time: int = 3,   
        # ETA gating. At long distances, require ETA <= eta_threshold to
        # avoid preempting for EMVs that are queued upstream and won't arrive
        # soon. Bypassed below always_preempt_within (EMV is essentially at
        # the stop line, preempt regardless of speed).
        eta_threshold_s: float = float('inf'),   # disabled by default
        always_preempt_within_m: float = 50.0,
    ):
        self.tid = tls_id
        self.detection_distance = float(detection_distance)
        self.eta_threshold_s = float(eta_threshold_s)
        self.always_preempt_within_m = float(always_preempt_within_m)
        self.release_distance = (
            float(release_distance) if release_distance is not None
            else self.detection_distance * 1.2
        )
        if self.release_distance < self.detection_distance:
            raise ValueError(
                f'release_distance ({self.release_distance}) must be >= '
                f'detection_distance ({self.detection_distance})'
            )
        if not (0.0 < detection_probability <= 1.0):
            raise ValueError('detection_probability must be in (0, 1]')
        self.detection_probability = float(detection_probability)
        self.comm_latency_steps = max(0, int(comm_latency_steps))
        self._rng = random.Random(
            rng_seed if rng_seed is not None else hash(tls_id) & 0xFFFFFFFF
        )
        self.debug = debug

        # Static program snapshot.
        logic = traci.trafficlight.getAllProgramLogics(self.tid)[0]
        self.phases = logic.phases

        # Dynamic state.
        self.active = False
        self._pending_action: int | None = None
        self._latency_countdown = 0
        # Remember EMVs we've already rolled a detection roll for, so the
        # detection_probability draw is made ONCE per EMV-arrival event, not
        # every tick (which would cause rapid flicker on/off).
        self._detected_vehicles: set[str] = set()
        self._failed_detection_vehicles: set[str] = set()

        # Rate-limited warning counters.
        self._warn_no_green_phases_count = 0
        self._warn_no_phase_for_link_count = 0
        self._warn_state_mismatch_count = 0

        # Raw-TraCI step() state machine (used by fixed/maxpressure runners):
        self.yellow_time = max(0, int(yellow_time))
        self._substate: str | None = None
        self._yellow_elapsed = 0
        self._current_target_state: str | None = None
        self._saved_program: str | None = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _emv_approaches(self) -> list[EMVApproach]:
        """Return EMVs approaching THIS TLS within the current threshold,
        nearest-first."""
        threshold = self.release_distance if self.active else self.detection_distance
        approaches: list[EMVApproach] = []

        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return approaches

        for vid in vehicle_ids:
            try:
                if traci.vehicle.getVehicleClass(vid) != 'emergency':
                    continue
                # Each entry is (tls_id, link_idx, distance, current_state),
                # ordered along the route. Only the first entry matching
                # this TLS is interesting to us.
                next_tls_list = traci.vehicle.getNextTLS(vid)
            except traci.TraCIException:
                continue

            for tls_id, link_idx, distance, _state in next_tls_list:
                if tls_id != self.tid:
                    continue
                if distance > threshold:
                    break
                if distance > self.always_preempt_within_m and self.eta_threshold_s != float('inf'):
                    try:
                        spd = traci.vehicle.getSpeed(vid)
                    except traci.TraCIException:
                        spd = 0.0
                    effective_speed = max(spd, 1.0)   # avoid div-by-zero for stopped EMVs
                    if distance / effective_speed > self.eta_threshold_s:
                        break

                # Detection dropout: decide once per vehicle per arrival,
                # not once per tick.
                if self.detection_probability < 1.0:
                    if vid in self._failed_detection_vehicles:
                        break
                    if vid not in self._detected_vehicles:
                        if self._rng.random() > self.detection_probability:
                            self._failed_detection_vehicles.add(vid)
                            break
                        self._detected_vehicles.add(vid)

                try:
                    spd = traci.vehicle.getSpeed(vid)
                except traci.TraCIException:
                    spd = 0.0

                approaches.append(EMVApproach(
                    vehicle_id=vid,
                    link_idx=int(link_idx),
                    distance=float(distance),
                    speed=float(spd),
                ))
                break  # take only the nearest entry for our TLS per vehicle

        approaches.sort(key=lambda a: a.distance)

        # Garbage-collect memory of vehicles that have left the sim so the
        # detection-state sets don't grow unbounded over long runs.
        if self._detected_vehicles or self._failed_detection_vehicles:
            live = set(vehicle_ids)
            self._detected_vehicles &= live
            self._failed_detection_vehicles &= live

        return approaches

    # ------------------------------------------------------------------
    # Phase selection
    # ------------------------------------------------------------------

    def _phase_state_for_link(self, link_idx: int) -> str | None:
        """State string of the first green-only phase that gives green at
        `link_idx`. Yellow-containing phases are skipped."""
        for phase in self.phases:
            state = phase.state
            if 'y' in state or 'Y' in state:
                continue
            if link_idx < len(state) and state[link_idx] in ('G', 'g'):
                return state
        return None

    def _current_state(self) -> str | None:
        try:
            return traci.trafficlight.getRedYellowGreenState(self.tid)
        except traci.TraCIException:
            return None

    def _current_state_greens_link(self, link_idx: int) -> bool:
        state = self._current_state()
        return (state is not None
                and link_idx < len(state)
                and state[link_idx] in ('G', 'g'))

    def _current_action_idx(self, green_phases) -> int | None:
        """Index in green_phases matching the current live state, or None
        if the current state is a yellow transition not in the green-phase
        list."""
        state = self._current_state()
        if state is None:
            return None
        for idx, ph in enumerate(green_phases):
            if getattr(ph, 'state', None) == state:
                return idx
        return None

    def _action_for_state(self, target_state: str, green_phases) -> int | None:
        """Map a target state string to an index in sumo-rl's green_phases.
        Exact match first; then a 'phase greens are a superset of target
        greens' fallback for case/padding mismatches."""
        for idx, ph in enumerate(green_phases):
            if getattr(ph, 'state', None) == target_state:
                return idx

        target_greens = {i for i, c in enumerate(target_state) if c in 'Gg'}
        if not target_greens:
            return None
        best_idx, best_overlap = None, 0
        for idx, ph in enumerate(green_phases):
            state = getattr(ph, 'state', '')
            if 'y' in state or 'Y' in state:
                continue
            phase_greens = {i for i, c in enumerate(state) if c in 'Gg'}
            if target_greens.issubset(phase_greens):
                overlap = len(target_greens & phase_greens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
        return best_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_preempt_action(self, traffic_signal) -> int | None:
        """Decide the preemption action for this tick.

        Returns:
            An int action index into sumo-rl's green_phases (returning the
            CURRENT index means 'hold / extend'), or None if no preemption
            is needed this tick.
        """
        approaches = self._emv_approaches()

        # --- Release: no EMV in range. ---
        if not approaches:
            if self.active:
                self.active = False
                self._pending_action = None
                self._latency_countdown = 0
                if self.debug:
                    print(f'[preempt] {self.tid}: RELEASE')
            return None

        green_phases = (
            getattr(traffic_signal, 'green_phases', None)
            or getattr(traffic_signal, 'all_phases', None)
        )
        if green_phases is None:
            self._warn('_warn_no_green_phases_count',
                       f'[preempt WARN] {self.tid}: no green_phases on signal')
            return None

        nearest = approaches[0]

        # --- Extension: if current phase already serves this EMV, hold. ---
        if self._current_state_greens_link(nearest.link_idx):
            hold = self._current_action_idx(green_phases)
            if hold is not None:
                self.active = True
                self._pending_action = None
                self._latency_countdown = 0
                if self.debug:
                    tail = ''
                    if len(approaches) > 1:
                        tail = (' + '
                                + ', '.join(f'{a.vehicle_id}@{a.distance:.0f}m'
                                            for a in approaches[1:]))
                    print(f'[preempt] {self.tid}: EXTEND '
                          f'{nearest.vehicle_id}@{nearest.distance:.0f}m '
                          f'link={nearest.link_idx}{tail}')
                return hold
            # Current state is a yellow transition — fall through to switch.

        # --- Switch: find the phase for the nearest EMV's link. ---
        target_state = self._phase_state_for_link(nearest.link_idx)
        if target_state is None:
            self._warn('_warn_no_phase_for_link_count',
                       f'[preempt WARN] {self.tid}: no green phase covers '
                       f'link {nearest.link_idx}')
            return None

        target_idx = self._action_for_state(target_state, green_phases)
        if target_idx is None:
            gp_states = [getattr(p, 'state', '<none>') for p in green_phases]
            self._warn('_warn_state_mismatch_count',
                       f'[preempt WARN] {self.tid}: state {target_state!r} '
                       f'not in green_phases {gp_states}')
            return None

        # --- Optional comm latency: delay the switch by N ticks. ---
        if self.comm_latency_steps > 0:
            if self._pending_action != target_idx:
                self._pending_action = target_idx
                self._latency_countdown = self.comm_latency_steps
            if self._latency_countdown > 0:
                self._latency_countdown -= 1
                # Stall — hold whatever phase is current for one more tick.
                return self._current_action_idx(green_phases)

        self.active = True
        self._pending_action = None
        if self.debug:
            print(f'[preempt] {self.tid}: SWITCH -> phase {target_idx} '
                  f'for {nearest.vehicle_id}@{nearest.distance:.0f}m '
                  f'link={nearest.link_idx}')
        return target_idx

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _warn(self, counter_name: str, msg: str, every: int = 20) -> None:
        if not self.debug:
            return
        count = getattr(self, counter_name, 0) + 1
        setattr(self, counter_name, count)
        if count == 1 or count % every == 0:
            print(f'{msg}  (occurrence #{count})')



    # ------------------------------------------------------------------
    # step() — raw-traci 4-state machine (UNUSED by current hybrid runner)
    # ------------------------------------------------------------------
    # Kept for a hypothetical future runner that manages SUMO itself via
    # traci and wants explicit yellow-in/yellow-out transitions per TLS.
    # The current hybrid runner delegates yellow insertion to sumo-rl.
    #
    # If you never plan to use raw-traci preemption, this block (and the
    # _apply_state / _release_program / _compute_*_yellow helpers below)
    # can be deleted without affecting the pipeline.
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Raw-TraCI preemption tick. For runners that manage SUMO directly
        (not via sumo-rl). Returns True while preemption is active in any
        substate, False otherwise.
    
        4-substate machine:
            inactive  --(EMV)-->  entering_yellow  --(yellow_time)-->  green
            green     --(gone)->  exiting_yellow   --(yellow_time)-->  inactive
            exiting_yellow --(EMV returns)--> green (abort release)
        """
        approaches = self._emv_approaches()
    
        target_state: str | None = None
        if approaches:
            nearest = approaches[0]
            if self._current_state_greens_link(nearest.link_idx):
                target_state = self._current_state()          # extension: hold
            else:
                target_state = self._phase_state_for_link(nearest.link_idx)
                if target_state is None:
                    self._warn('_warn_no_phase_for_link_count',
                               f'[preempt WARN] {self.tid}: no green phase '
                               f'covers link {nearest.link_idx}')
    
        # ---------- Release path ----------
        if target_state is None:
            if not self.active:
                return False
            if self._substate in ('green', 'entering_yellow'):
                yellow_state = self._compute_release_yellow(self._current_target_state)
                if yellow_state is not None and self.yellow_time > 0:
                    self._apply_state(yellow_state)
                    self._substate = 'exiting_yellow'
                    self._yellow_elapsed = 0
                    return True
                self._release_program()
                return False
            # substate == 'exiting_yellow'
            self._yellow_elapsed += 1
            if self._yellow_elapsed < self.yellow_time:
                return True
            self._release_program()
            return False
    
        # ---------- Preempt path ----------
        if not self.active:
            try:
                self._saved_program = traci.trafficlight.getProgram(self.tid)
                current_state = traci.trafficlight.getRedYellowGreenState(self.tid)
            except traci.TraCIException:
                self._saved_program = None
                current_state = ''
            self.active = True
            self._current_target_state = target_state
            yellow_state = self._compute_entry_yellow(current_state, target_state)
            if yellow_state is not None and self.yellow_time > 0:
                self._apply_state(yellow_state)
                self._substate = 'entering_yellow'
                self._yellow_elapsed = 0
            else:
                self._apply_state(target_state)
                self._substate = 'green'
            if self.debug:
                print(f'[preempt] {self.tid}: ENTER -> {approaches[0].vehicle_id}'
                      f'@{approaches[0].distance:.0f}m link={approaches[0].link_idx}')
            return True
    
        if self._substate == 'entering_yellow':
            self._yellow_elapsed += 1
            if self._yellow_elapsed >= self.yellow_time:
                self._apply_state(target_state)
                self._current_target_state = target_state
                self._substate = 'green'
            return True
    
        if self._substate == 'exiting_yellow':
            # EMV reappeared mid-release — abort, cut straight to green.
            self._apply_state(target_state)
            self._current_target_state = target_state
            self._substate = 'green'
            return True
    
        # substate == 'green' — check for retarget (different EMV, different link)
        if target_state != self._current_target_state:
            yellow_state = self._compute_entry_yellow(
                self._current_target_state, target_state
            )
            if yellow_state is not None and self.yellow_time > 0:
                self._apply_state(yellow_state)
                self._substate = 'entering_yellow'
                self._yellow_elapsed = 0
                self._current_target_state = target_state
            else:
                self._apply_state(target_state)
                self._current_target_state = target_state
            return True
    
        # Stable green — re-assert each tick so the underlying program can't drift back.
        self._apply_state(target_state)
        return True

    # ------------------------------------------------------------------
    # step() helpers (also unused by hybrid runner)
    # ------------------------------------------------------------------

    def _apply_state(self, state: str | None) -> None:
        if not state:
            return
        try:
            traci.trafficlight.setRedYellowGreenState(self.tid, state)
        except traci.TraCIException:
            pass

    def _release_program(self) -> None:
        self.active = False
        self._substate = None
        self._yellow_elapsed = 0
        self._current_target_state = None
        if self._saved_program is not None:
            try:
                traci.trafficlight.setProgram(self.tid, self._saved_program)
                traci.trafficlight.setPhase(self.tid, 0)
            except traci.TraCIException:
                pass
            self._saved_program = None

    def _compute_entry_yellow(self, current_state: str,
                              target_state: str) -> str | None:
        if not current_state or len(current_state) != len(target_state):
            return None
        out = []
        needs_yellow = False
        for c, t in zip(current_state, target_state):
            if c in 'Gg' and t not in 'Gg':
                out.append('y'); needs_yellow = True
            elif c in 'Gg' and t in 'Gg':
                out.append('G')
            else:
                out.append('r')
        return ''.join(out) if needs_yellow else None

    def _compute_release_yellow(self,
                                active_state: str | None) -> str | None:
        if not active_state:
            return None
        return ''.join('y' if c in 'Gg' else 'r' for c in active_state)


# Add this to wherever emergeny preemption is called
# #EmergencyPreemption(
#     tid,
#     detection_distance=250.0,
#     release_distance=300.0,
#     eta_threshold_s=25.0,        # don't preempt for EMVs >25s away
#     always_preempt_within_m=50.0,
#     debug=debug_preempt,
# )