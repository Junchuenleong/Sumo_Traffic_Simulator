import numpy as np
import traci
from gymnasium.spaces import Box
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class ObsRewardWrapper(BaseParallelWrapper):
    """
    PettingZoo-compatible wrapper for SUMO-RL parallel_env that:
      1. augments each agent observation with neighbor + EMV features
      2. shapes the base reward with optional penalties

    EMV features are computed VEHICLE-CENTRICALLY using
    traci.vehicle.getNextTLS(), so detection is:
      * route-aware (a nearby EMV that will NOT pass through this TLS
        is correctly ignored),
      * measured in meters-along-route to the stop line (not meters of
        current-lane), which matters when TLS-controlled lanes are short
        stubs after OSM --tls.join (we saw 0.2-14m on this net).
    """

    EXTRA_OBS_DIM = 7   # 4 neighbor features + 3 EMV features

    def __init__(
        self,
        env,
        neighbor_map=None,
        emv_detection_distance: float = 250.0,
        spillback_threshold: float = 0.80,
        switch_penalty_coef: float = 0.05,
        spillback_penalty_coef: float = 0.00,
        emv_stop_penalty_coef: float = 2.00,
    ):
        super().__init__(env)
        self.neighbor_map = neighbor_map or {}
        self.emv_detection_distance = float(emv_detection_distance)
        self.spillback_threshold = float(spillback_threshold)
        self.switch_penalty_coef = float(switch_penalty_coef)
        self.spillback_penalty_coef = float(spillback_penalty_coef)
        self.emv_stop_penalty_coef = float(emv_stop_penalty_coef)
        self._agent_index = {}  # fill on first reset
        self.last_actions = {}

    # ------------------------------------------------------------------
    # Unchanged helpers
    # ------------------------------------------------------------------

    def _get_identity_features(self, tls_id):
        try:
            n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        except traci.TraCIException:
            n_phases = 0
        idx = self._agent_index.get(tls_id, 0)
        n_agents = max(1, len(self._agent_index))
        return np.array([n_phases / 10.0, idx / n_agents], dtype=np.float32)

    def observation_space(self, agent):
        base_space = self.env.observation_space(agent)

        if not isinstance(base_space, Box):
            raise TypeError(
                f"ObsRewardWrapper expects Box observation spaces, got "
                f"{type(base_space)} for agent {agent}."
            )

        base_dim = int(np.prod(base_space.shape))
        total_dim = base_dim + self.EXTRA_OBS_DIM

        low = np.full(total_dim, -np.inf, dtype=np.float32)
        high = np.full(total_dim, np.inf, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.last_actions = {}

        result = self.env.reset(seed=seed, options=options)

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._augment_obs_dict(obs), info

        return self._augment_obs_dict(result)

    def step(self, actions):
        result = self.env.step(actions)

        if len(result) == 5:
            obs, rewards, terminations, truncations, infos = result
            aug_obs = self._augment_obs_dict(obs)
            shaped_rewards = self._shape_rewards(rewards, actions)
            self.last_actions = dict(actions)
            return aug_obs, shaped_rewards, terminations, truncations, infos

        if len(result) == 4:
            obs, rewards, dones, infos = result
            aug_obs = self._augment_obs_dict(obs)
            shaped_rewards = self._shape_rewards(rewards, actions)
            self.last_actions = dict(actions)
            return aug_obs, shaped_rewards, dones, infos

        raise ValueError(
            f"Unsupported step() return format with {len(result)} items."
        )

    def _augment_obs_dict(self, obs_dict):
        new_obs = {}
        for agent_id, obs in obs_dict.items():
            base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            neighbor_feats = self._get_neighbor_features(agent_id)
            emv_feats = self._get_emv_features(agent_id)
            extra_feats = np.concatenate([neighbor_feats, emv_feats]).astype(np.float32)
            new_obs[agent_id] = np.concatenate([base_obs, extra_feats]).astype(np.float32)
        return new_obs

    def _get_neighbor_features(self, tls_id: str) -> np.ndarray:
        local_in = self._sum_incoming_halting(tls_id)
        local_out = self._sum_outgoing_halting(tls_id)

        neighbors = self.neighbor_map.get(tls_id, [])
        neighbor_in_sum = 0.0
        for nid in neighbors:
            neighbor_in_sum += self._sum_incoming_halting(nid)

        return np.array(
            [local_in, local_out, neighbor_in_sum, float(len(neighbors))],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # EMV detection — VEHICLE-BASED via getNextTLS (new)
    # ------------------------------------------------------------------

    def _emv_approaches_for_tls(self, tls_id: str):
        """Return a list of (distance_m, speed_mps) for all EMVs whose
        route includes this TLS and whose distance to the stop line is
        within emv_detection_distance. List is unsorted; callers that
        want 'nearest first' should sort by the first element.

        Route-awareness means: if an EMV is currently on a lane physically
        near this TLS but will turn onto a different edge before reaching
        it, the EMV is correctly *not* returned."""
        out = []
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return out

        for vid in vehicle_ids:
            try:
                if traci.vehicle.getVehicleClass(vid) != "emergency":
                    continue
                # Each entry: (tls_id, link_idx, distance, current_state),
                # ordered along the vehicle's route.
                next_tls = traci.vehicle.getNextTLS(vid)
            except traci.TraCIException:
                continue

            for tid, _link_idx, dist, _state in next_tls:
                if tid != tls_id:
                    continue
                if dist > self.emv_detection_distance:
                    break
                try:
                    spd = traci.vehicle.getSpeed(vid)
                except traci.TraCIException:
                    spd = 0.0
                out.append((float(dist), float(spd)))
                break  # only the first (nearest) entry for our TLS

        return out

    def _get_emv_features(self, tls_id: str) -> np.ndarray:
        """3 features fed to PPO: (emv_present, dist_norm, emv_stopped).

        dist_norm in [0,1]: 0 == at stop line, 1 == at/beyond detection
        distance or no EMV. Same contract as before so the observation
        space and any saved VecNormalize stats stay compatible."""
        approaches = self._emv_approaches_for_tls(tls_id)
        if not approaches:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Nearest EMV drives the features.
        nearest_dist, nearest_speed = min(approaches, key=lambda t: t[0])

        dist_norm = float(np.clip(
            nearest_dist / max(self.emv_detection_distance, 1e-6),
            0.0, 1.0,
        ))
        emv_stopped = 1.0 if nearest_speed < 0.1 else 0.0
        return np.array([1.0, dist_norm, emv_stopped], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_rewards(self, reward_dict, actions):
        shaped = {}

        for tls_id, base_reward in reward_dict.items():
            r = float(base_reward)

            if self.spillback_penalty_coef != 0.0:
                r -= self.spillback_penalty_coef * self._spillback_penalty(tls_id)

            if self.emv_stop_penalty_coef != 0.0:
                r -= self.emv_stop_penalty_coef * self._emv_stop_penalty(tls_id)

            if self.switch_penalty_coef != 0.0:
                r -= self.switch_penalty_coef * self._switch_penalty(tls_id, actions)

            shaped[tls_id] = r

        return shaped

    def _spillback_penalty(self, tls_id: str) -> float:
        total = 0.0
        controlled_links = self._safe_get_controlled_links(tls_id)
        seen_out_lanes = set()

        for link_group in controlled_links:
            for (_in_lane, out_lane, _via) in link_group:
                if not out_lane or out_lane in seen_out_lanes:
                    continue
                seen_out_lanes.add(out_lane)

                try:
                    occ = traci.lane.getLastStepOccupancy(out_lane) / 100.0
                except traci.TraCIException:
                    continue

                if occ > self.spillback_threshold:
                    total += (occ - self.spillback_threshold)

        return float(total)

    def _emv_stop_penalty(self, tls_id: str) -> float:
        """Count EMVs approaching this TLS that are stopped (speed < 0.1
        m/s). Uses the same route-aware detection as _get_emv_features,
        so an EMV stopped 200m upstream is still counted — which is what
        we want, because the penalty's job is to discourage situations
        where any inbound EMV is forced to a halt."""
        approaches = self._emv_approaches_for_tls(tls_id)
        return float(sum(1 for _dist, spd in approaches if spd < 0.1))

    def _switch_penalty(self, tls_id: str, actions) -> float:
        if tls_id not in actions or tls_id not in self.last_actions:
            return 0.0

        prev_action = self.last_actions[tls_id]
        curr_action = actions[tls_id]

        try:
            changed = not np.array_equal(
                np.asarray(prev_action), np.asarray(curr_action)
            )
        except Exception:
            changed = prev_action != curr_action

        return 1.0 if changed else 0.0

    # ------------------------------------------------------------------
    # Low-level helpers (unchanged)
    # ------------------------------------------------------------------

    def _get_controlled_lanes(self, tls_id: str):
        try:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
        except traci.TraCIException:
            return []
        return list(dict.fromkeys(lanes))

    def _safe_get_controlled_links(self, tls_id: str):
        try:
            return traci.trafficlight.getControlledLinks(tls_id)
        except traci.TraCIException:
            return []

    def _sum_incoming_halting(self, tls_id: str) -> float:
        total = 0.0
        for lane_id in self._get_controlled_lanes(tls_id):
            try:
                total += traci.lane.getLastStepHaltingNumber(lane_id)
            except traci.TraCIException:
                continue
        return float(total)

    def _sum_outgoing_halting(self, tls_id: str) -> float:
        total = 0.0
        seen_out_lanes = set()

        for link_group in self._safe_get_controlled_links(tls_id):
            for (_in_lane, out_lane, _via) in link_group:
                if not out_lane or out_lane in seen_out_lanes:
                    continue
                seen_out_lanes.add(out_lane)

                try:
                    total += traci.lane.getLastStepHaltingNumber(out_lane)
                except traci.TraCIException:
                    continue

        return float(total)