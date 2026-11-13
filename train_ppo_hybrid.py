"""
Hybrid PPO control:
    - PPO handles normal traffic
    - EmergencyPreemption overrides control when an EMV is close

Operates at the PettingZoo parallel-env level (dict obs / dict actions)
so that per-agent preemption decisions can override PPO on a per-TLS basis.

Usage:
    python train_ppo_hybrid.py                              # default (offpeak + EMV)
    python run.py hybrid --scenario offpeak --emv           # via unified CLI
    python run.py hybrid --scenario rushhour --emv --model models/best/best_model
    python run.py hybrid --scenario offpeak --emv --inspect-emv   # EMV-route diagnostic
    python train_ppo_hybrid.py --gui                        # run with sumo gui
"""

import csv
import os
import pickle

import numpy as np
import sumo_setup  # noqa: F401
import sumo_rl
import supersuit as ss
import traci
from stable_baselines3 import PPO

from config import Config, EmergencyPreemption, ScenarioConfig
from neighbour_map import build_neighbor_map
from obs_reward_wrapper import ObsRewardWrapper
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary


# ---------------------------------------------------------------------------
# VecNormalize stats loader (matches PPO training)
# ---------------------------------------------------------------------------

def load_obs_rms(path: str):
    with open(path, 'rb') as f:
        vn = pickle.load(f)
    return vn.obs_rms, vn.clip_obs, vn.epsilon


def normalize_obs(obs, obs_rms, clip_obs, epsilon):
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
        -clip_obs, clip_obs,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Env builder — PettingZoo level (NO VecEnv conversion)
# ---------------------------------------------------------------------------

def make_pz_env(cfg: Config, seed: int = 0, gui: bool = False):
    neighbor_map = build_neighbor_map(cfg.network.net_file)
    out_prefix = cfg.out_prefix("ppo_hybrid")

    env = sumo_rl.parallel_env(
        net_file=cfg.network.net_file,
        route_file=cfg.route_files_str(),
        out_csv_name=out_prefix,
        use_gui=gui,
        num_seconds=cfg.sim.num_seconds,
        delta_time=cfg.sim.delta_time,
        yellow_time=cfg.sim.yellow_time,
        min_green=cfg.sim.min_green,
        max_green=cfg.sim.max_green,
        reward_fn='pressure',
        sumo_seed=seed,
        time_to_teleport=cfg.sim.time_to_teleport,
        additional_sumo_cmd=build_sumo_cmd(
            out_prefix=out_prefix,
            log_level='quiet',
        ),
    )
    if not hasattr(env.unwrapped, 'render_mode'):
        env.unwrapped.render_mode = None

    env = ObsRewardWrapper(
        env,
        neighbor_map=neighbor_map,
        spillback_penalty_coef=cfg.reward.spillback_penalty_coef,
        switch_penalty_coef=cfg.reward.switch_penalty_coef,
        emv_stop_penalty_coef=cfg.reward.emv_stop_penalty_coef,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    return env


# ---------------------------------------------------------------------------
# Traffic-signal introspection
# ---------------------------------------------------------------------------

def get_traffic_signals(env):
    e = env
    for _ in range(10):
        if hasattr(e, 'traffic_signals'):
            return e.traffic_signals
        e = getattr(e, 'env', getattr(e, 'unwrapped', None))
        if e is None:
            break
    raise RuntimeError('Could not locate sumo_rl traffic_signals on env')


def get_native_action_n(traffic_signal) -> int:
    green_phases = (
        getattr(traffic_signal, 'green_phases', None)
        or getattr(traffic_signal, 'all_phases', None)
    )
    return len(green_phases) if green_phases is not None else 1


# ---------------------------------------------------------------------------
# EMV-route diagnostics (optional)
# ---------------------------------------------------------------------------

def inspect_emv_coverage(tls_ids):
    """Print which TLSs each EMV will traverse. Useful for confirming that
    the preemption controllers will actually fire on these ambulance routes.

    If an EMV's route doesn't touch any controlled edge, preemption can
    never activate for it — check your EMV route definitions in that case.
    """
    tls_edges = {}
    for tid in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tid)
        tls_edges[tid] = {lane.rsplit('_', 1)[0] for lane in lanes}
        print(f'  TLS {tid} controls {len(lanes)} lanes '
              f'on {len(tls_edges[tid])} edges')

    pending = list(traci.simulation.getPendingVehicles())
    loaded = list(traci.vehicle.getIDList())
    emv_any = False
    for vid in pending + loaded:
        try:
            if traci.vehicle.getVehicleClass(vid) != 'emergency':
                continue
        except traci.TraCIException:
            continue
        emv_any = True
        route = traci.vehicle.getRoute(vid)
        hits = {tid: (tls_edges[tid] & set(route)) for tid in tls_ids}
        hits = {k: sorted(v) for k, v in hits.items() if v}
        print(f'  EMV {vid}: {len(route)} edges, {route[0]} -> {route[-1]}')
        if hits:
            for tid, edges in hits.items():
                print(f'       via TLS {tid}: {edges}')
        else:
            print(f'       ⚠ route crosses NO controlled edge — '
                  f'preemption will never fire')

    if not emv_any:
        print('  No emergency vehicles found in the loaded/pending lists.')


# ---------------------------------------------------------------------------
# Main hybrid loop
# ---------------------------------------------------------------------------

def run_hybrid(cfg: Config, model_path: str | None = None,
               gui: bool = False, seed: int = 0,
               debug_preempt: bool = False, inspect_emv: bool = False):
    if model_path is None:
        model_path = cfg.model_path('ppo_shared_final')

    model_file = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Model file not found: {model_file}')

    vecnorm_path = os.path.join(cfg.models_dir, 'vecnormalize.pkl')
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f'VecNormalize stats not found at {vecnorm_path}. '
            'Re-run train first:  python run.py train'
        )

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    out_prefix = cfg.out_prefix("ppo_hybrid")

    print(f'Loading model: {model_file}')
    model = PPO.load(model_path, device='cpu')
    obs_rms, clip_obs, eps = load_obs_rms(vecnorm_path)

    env = make_pz_env(cfg, gui=gui, seed=seed)

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    tls_ids = list(env.possible_agents)
    print(f'Hybrid PPO + EMV preemption across {len(tls_ids)} TLS agents.')
    print(f'  Scenario        : {cfg.scenario.name}  (EMV={cfg.scenario.emv})')
    print(f'  Model obs shape : {model.observation_space.shape}')
    print(f'  Model act space : {model.action_space}')

    # Preemption controllers (needs TraCI up, which it is after reset)
    # All knobs come from cfg.preemption, but allow per-run debug override.
    preemption = {
        tid: EmergencyPreemption(
            tid,
            detection_distance=cfg.preemption.detection_distance,
            release_distance=cfg.preemption.release_distance,
            eta_threshold_s=cfg.preemption.eta_threshold_s,
            always_preempt_within_m=cfg.preemption.always_preempt_within_m,
            detection_probability=cfg.preemption.detection_probability,
            comm_latency_steps=cfg.preemption.comm_latency_steps,
            debug=debug_preempt or cfg.preemption.debug,
        )
        for tid in tls_ids
    }

    traffic_signals = get_traffic_signals(env)
    native_action_n = {
        tid: get_native_action_n(traffic_signals.get(tid))
        for tid in tls_ids
    }
    print(f'  Native action sizes: {native_action_n}')

    if inspect_emv:
        print('\n--- EMV route diagnostic ---')
        inspect_emv_coverage(tls_ids)
        print('--- end diagnostic ---\n')

    num_seconds = cfg.sim.num_seconds
    delta_time  = cfg.sim.delta_time
    steps_per_episode = num_seconds // delta_time

    phase_log = []  # local: (sim_time, tls_id, phase_index, controller)
    total_rewards = {tid: 0.0 for tid in tls_ids}
    preempt_cumulative = 0
    preempt_active_steps = {tid: 0 for tid in tls_ids}
    last_preempt_state = {tid: False for tid in tls_ids}
    clamp_counts = {tid: 0 for tid in tls_ids}

    for step in range(steps_per_episode):
        current_agents = list(env.agents)
        if not current_agents:
            break

        actions = {}
        n_active_preempts = 0
        newly_preempted = []
        newly_released = []

        for tid in current_agents:
            padded_n = env.action_space(tid).n
            native_n = native_action_n.get(tid, padded_n)
            max_valid = max(1, min(padded_n, native_n)) - 1

            # 1) Check preemption first
            ts = traffic_signals.get(tid)
            preempt_act = None
            if ts is not None:
                preempt_act = preemption[tid].get_preempt_action(ts)

            if preempt_act is not None:
                actions[tid] = min(int(preempt_act), max_valid)
                n_active_preempts += 1
                preempt_active_steps[tid] += 1
                if not last_preempt_state[tid]:
                    newly_preempted.append(tid)
                    last_preempt_state[tid] = True
                continue

            if last_preempt_state[tid]:
                newly_released.append(tid)
                last_preempt_state[tid] = False

            # 2) PPO inference
            if tid not in obs:
                actions[tid] = 0
                continue

            single_obs = np.asarray(obs[tid], dtype=np.float32).reshape(-1)
            norm = normalize_obs(single_obs, obs_rms, clip_obs, eps)
            act, _ = model.predict(norm, deterministic=True)
            if hasattr(act, 'item'):
                act = act.item()
            act = int(act)

            if act > max_valid:
                clamp_counts[tid] += 1
                act = max_valid
            actions[tid] = act

        # Report preemption transitions as they happen
        sim_t = (step + 1) * delta_time
        for tid in newly_preempted:
            print(f'  [{sim_t:>5}s] PREEMPT on   {tid}')
        for tid in newly_released:
            print(f'  [{sim_t:>5}s] PREEMPT off  {tid}')

        # Step the env
        result = env.step(actions)
        for tid in current_agents:
            phase_log.append((
                traci.simulation.getTime(),
                tid,
                traci.trafficlight.getPhase(tid),
                'preempt' if last_preempt_state[tid] else 'ppo',
            ))

        if len(result) == 5:
            obs, rewards, terms, truncs, infos = result
            done_all = all(
                terms.get(a, False) or truncs.get(a, False)
                for a in current_agents
            )
        else:
            obs, rewards, dones, infos = result
            done_all = all(dones.get(a, False) for a in current_agents)

        for tid, r in rewards.items():
            total_rewards[tid] = total_rewards.get(tid, 0.0) + float(r)
        preempt_cumulative += n_active_preempts

        if (step + 1) % 100 == 0:
            total_pa = sum(preempt_active_steps.values())
            print(f'  sim {sim_t:>5}s / {num_seconds}s  '
                  f'| now preempting: {n_active_preempts}  '
                  f'| cum preempt-steps: {total_pa}')

        if done_all:
            break

    env.close()

    # Phase log per-scenario so runs don't clobber each other
    log_path = f'{cfg.outputs_dir}/phase_log_ppo_{cfg.scenario.name}' \
               f'{"_emv" if cfg.scenario.emv else ""}.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerows([('t', 'tls', 'phase', 'ctrl'), *phase_log])

    mean_reward = sum(total_rewards.values()) / max(len(total_rewards), 1)
    print(f'\nHybrid run complete.')
    print(f'  Total preempt-steps        : {preempt_cumulative}')
    print(f'  Preempt-steps per TLS      : {preempt_active_steps}')
    if any(clamp_counts.values()):
        print(f'  Action clamps per TLS      : {clamp_counts}')
    print(f'  Mean cumul. reward per TLS : {mean_reward:.3f}')
    print(f'  Tripinfo                   : {out_prefix}_tripinfo.xml')
    print(f'  Emissions                  : {out_prefix}_emissions.xml')
    print(f'  Phase log                  : {log_path}')

    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='ppo_hybrid')


if __name__ == '__main__':
    # Standalone default: offpeak + EMV.
    # Use `python run.py hybrid --scenario ... [--emv]` for CLI overrides.
    run_hybrid(Config(scenario=ScenarioConfig(name="offpeak", emv=True)),
               gui=False, seed=0)