"""
Evaluate the trained PPO policy. Loads the final saved model, runs one full
episode with deterministic actions, and writes SUMO's tripinfo + emissions XMLs
so evaluate.py can compare all methods.

Usage:
    python run_ppo_eval.py                          # uses default Config()
    python run.py ppo --scenario offpeak            # via unified CLI
    python run.py ppo --scenario rushhour --emv
    python run.py ppo --scenario offpeak --model models/best/best_model
"""

import hashlib
import json
import os
import pickle
import sys

import numpy as np
import sumo_setup  # noqa: F401
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO

from config import Config
from neighbour_map import build_neighbor_map
from obs_reward_wrapper import ObsRewardWrapper
from sumo_cmd import build_sumo_cmd, print_tripinfo_summary


# ---------------------------------------------------------------------------
# VecNormalize observation stats loader
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

def make_eval_env(cfg: Config, seed: int = 0, gui: bool = False):
    neighbor_map = build_neighbor_map(cfg.network.net_file)
    out_prefix = cfg.out_prefix("ppo_eval")

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
# Traffic-signal introspection (for action clamping)
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
# Manifest check (detect train/eval drift)
# ---------------------------------------------------------------------------

def check_manifest(cfg: Config, model_path: str) -> None:
    manifest_path = f'{model_path}.manifest.json'
    if not os.path.exists(manifest_path):
        print(f'[info] no manifest found at {manifest_path} — skipping check')
        return

    with open(cfg.network.net_file, 'rb') as f:
        current_net_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    with open(manifest_path) as f:
        m = json.load(f)

    if m.get('net_sha256_prefix') != current_net_hash:
        print(f'[WARN] Network hash drift:')
        print(f'       model trained on net: {m.get("net_sha256_prefix")}')
        print(f'       current net:          {current_net_hash}')
        print(f'       Obs-space or TLS topology may not match — expect errors.')

    route_str = cfg.route_files_str()
    if m.get('route_file') != route_str:
        print(f'[info] Route file differs from training '
              f'({m.get("route_file")} vs {route_str}) — expected for evaluation.')


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: Config, model_path: str | None = None,
             gui: bool = False, seed: int = 0):
    if model_path is None:
        model_path = cfg.model_path("ppo_shared_final")

    model_file = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not os.path.exists(model_file):
        print(f'ERROR: Model file not found: {model_file}')
        sys.exit(1)

    vecnorm_path = os.path.join(cfg.models_dir, 'vecnormalize.pkl')
    if not os.path.exists(vecnorm_path):
        print(f'ERROR: VecNormalize stats not found at {vecnorm_path}')
        print('       Re-run train first:  python run.py train')
        sys.exit(1)

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    out_prefix = cfg.out_prefix("ppo_eval")

    check_manifest(cfg, model_path)

    print(f'Loading model: {model_file}')
    model = PPO.load(model_path, device='cpu')
    obs_rms, clip_obs, eps = load_obs_rms(vecnorm_path)

    env = make_eval_env(cfg, gui=gui, seed=seed)

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    tls_ids = list(env.possible_agents)
    traffic_signals = get_traffic_signals(env)
    native_action_n = {
        tid: get_native_action_n(traffic_signals.get(tid))
        for tid in tls_ids
    }

    num_seconds = cfg.sim.num_seconds
    delta_time  = cfg.sim.delta_time
    print(f'Evaluating PPO across {len(tls_ids)} TLS agents '
          f'for {num_seconds}s ({num_seconds // delta_time} decision steps)...')
    print(f'  Scenario            : {cfg.scenario.name}  (EMV={cfg.scenario.emv})')
    print(f'  Model obs shape     : {model.observation_space.shape}')
    print(f'  Model action space  : {model.action_space}')
    print(f'  Native action sizes : {native_action_n}')
    if gui:
        print('  [GUI mode] SUMO window will open — press Play in the toolbar.')

    steps_per_episode = num_seconds // delta_time
    total_reward = 0.0
    clamp_counts = {tid: 0 for tid in tls_ids}
    step = 0

    for step in range(steps_per_episode - 1):
        current_agents = list(env.agents)
        if not current_agents:
            break

        actions = {}
        for tid in current_agents:
            padded_n = env.action_space(tid).n
            native_n = native_action_n.get(tid, padded_n)
            max_valid = max(1, min(padded_n, native_n)) - 1

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

        result = env.step(actions)
        if len(result) == 5:
            obs, rewards, terms, truncs, infos = result
            done_all = all(
                terms.get(a, False) or truncs.get(a, False)
                for a in current_agents
            )
        else:
            obs, rewards, dones, infos = result
            done_all = all(dones.get(a, False) for a in current_agents)

        total_reward += float(np.mean(list(rewards.values()))) if rewards else 0.0

        if (step + 1) % 100 == 0:
            elapsed_sim = (step + 1) * delta_time
            print(f'  sim {elapsed_sim:>5}s / {num_seconds}s  '
                  f'| avg reward so far: {total_reward / (step + 1):.4f}')

        if done_all:
            break

    env.close()

    print(f'\nEvaluation complete.')
    print(f'  Mean reward per step : {total_reward / max(step + 1, 1):.4f}')
    if any(clamp_counts.values()):
        print(f'  Action clamps        : {clamp_counts}')
        print('  (This is expected when TLSs have different green-phase counts.)')
    print(f'  Tripinfo             : {out_prefix}_tripinfo.xml')
    print(f'  Emissions            : {out_prefix}_emissions.xml')

    print_tripinfo_summary(f'{out_prefix}_tripinfo.xml', label='ppo_eval')

    print(f'\nNow run:  python run.py evaluate')


if __name__ == '__main__':
    # Standalone default: offpeak + EMV (the scenario the user trained on).
    # Use `python run.py ppo --scenario ... [--emv]` for CLI overrides.
    from config import ScenarioConfig
    evaluate(Config(scenario=ScenarioConfig(name="offpeak", emv=True)),
             gui=False, seed=0)