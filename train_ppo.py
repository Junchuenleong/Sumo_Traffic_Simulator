"""
Multi-agent PPO with parameter sharing for traffic signal control.

Usage:
    python train_ppo.py                                     # default Config()
    python run.py train --scenario offpeak --emv            # recommended (EMV features see non-zero data)
    python run.py train --scenario offpeak --emv --timesteps 1000000 --seed 42
    python train_ppo.py --sanity                            # one-off TLS phase-count check

Note: train with --emv. If you train on plain offpeak (no EMV), the three EMV
observation features stay at their all-zeros default and VecNormalize learns
near-zero variance for them. At hybrid-eval time, real EMVs produce non-zero
values that then explode after normalization, and the policy behaves badly.
"""

import argparse
import hashlib
import json
import os

import sumo_setup  # noqa: F401  must run before importing sumo_rl/traci
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, sync_envs_normalization

import torch
torch.set_num_threads(os.cpu_count() or 1)

from config import Config, ScenarioConfig
from neighbour_map import build_neighbor_map
from obs_reward_wrapper import ObsRewardWrapper
from sumo_cmd import build_sumo_cmd


class SyncingEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize running stats from the training
    env into the eval env immediately before each evaluation. Without this,
    eval observes un-normalized data while the policy was trained on
    normalized data, and reported eval reward is junk."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                # Neither env is wrapped in VecNormalize — nothing to sync.
                pass
        return super()._on_step()


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def make_env(cfg: Config, seed: int = 0, gui: bool = False,
             out_tag: str = 'train') -> object:
    """Build a vectorised multi-agent env for PPO training."""
    neighbor_map = build_neighbor_map(cfg.network.net_file)

    env = sumo_rl.parallel_env(
        net_file=cfg.network.net_file,
        route_file=cfg.route_files_str(),
        out_csv_name=f'{cfg.outputs_dir}/ppo_{out_tag}',
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
            out_prefix=None,     # don't write tripinfo during training
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
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1,
                                base_class='stable_baselines3')

    # SuperSuit's ConcatVecEnv doesn't expose .seed(), which newer SB3 calls
    # from set_random_seed(). SUMO seeding is already handled via sumo_seed=
    # above; add a contract-compliant no-op so the wrapper chain doesn't break.
    env.seed = lambda s=None: [s] * env.num_envs

    return env


# ---------------------------------------------------------------------------
# Sanity check (run once, then never again)
# ---------------------------------------------------------------------------

def sanity_check_tls(cfg: Config) -> None:
    """Print phase counts per TLS. Run once to confirm the net's TLS programs
    look right and the action-space padding isn't wildly uneven across agents."""
    import traci
    env = make_env(cfg, seed=0, gui=False, out_tag='sanity')
    env.reset()

    tls_ids = traci.trafficlight.getIDList()
    print(f'\n{"=" * 60}')
    print(f'Found {len(tls_ids)} traffic lights')
    print(f'{"=" * 60}')

    green_counts = []
    for tid in tls_ids:
        try:
            logic = traci.trafficlight.getAllProgramLogics(tid)[0]
            phases = logic.phases
            states = [p.state for p in phases]
            durations = [p.duration for p in phases]
            green_phases = [s for s in states
                            if 'y' not in s and 'Y' not in s]
            green_counts.append(len(green_phases))
            print(f'  {tid}: {len(phases)} phases ({len(green_phases)} green) | '
                  f'durs={durations}')
        except traci.TraCIException as e:
            print(f'  {tid}: ERROR {e}')

    if green_counts:
        spread = max(green_counts) - min(green_counts)
        print(f'\nGreen-phase distribution: min={min(green_counts)}, '
              f'max={max(green_counts)} (spread={spread})')
        if spread > 2:
            print(f'⚠ Large spread ({spread}) in green-phase counts across TLSs.')
            print(f'  After padding, PPO\'s action space is Discrete({max(green_counts)}).')
            print(f'  On the smaller TLS, actions >= {min(green_counts)} will be clamped.')

    env.close()


# ---------------------------------------------------------------------------
# Manifest (helps eval detect train/eval drift)
# ---------------------------------------------------------------------------

def write_manifest(cfg: Config, env, model_path: str) -> None:
    with open(cfg.network.net_file, 'rb') as f:
        net_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    try:
        obs_shape = list(env.observation_space.shape)
    except Exception:
        obs_shape = None
    try:
        action_n = int(env.action_space.n)
    except Exception:
        action_n = None

    manifest = {
        'net_file': cfg.network.net_file,
        'net_sha256_prefix': net_hash,
        'route_file': cfg.route_files_str(),
        'obs_shape': obs_shape,
        'action_n': action_n,
        'num_timesteps': cfg.train.total_timesteps,
        'delta_time': cfg.sim.delta_time,
        'num_seconds': cfg.sim.num_seconds,
    }
    path = f'{model_path}.manifest.json'
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest written: {path}')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    seed = cfg.train.seed

    # Train env
    env = make_env(cfg, seed=seed, gui=False, out_tag='train')
    env = VecMonitor(env)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=cfg.train.gamma,
    )

    # Eval env — different sumo_seed so eval isn't just memorising the
    # training seed's traffic pattern.
    eval_env = make_env(cfg, seed=seed + 1000, gui=False, out_tag='train_eval')
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=cfg.train.gamma, training=False,
    )

    print(f'Parameter sharing: {env.num_envs} TLS agents -> one shared PPO policy.')
    print(f'Scenario        : {cfg.scenario.name}  (EMV={cfg.scenario.emv})')
    print(f'Train route     : {cfg.route_files_str()}')
    print(f'Total timesteps : {cfg.train.total_timesteps:,}')
    print(f'Seed            : {seed}')

    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        verbose=1,
        tensorboard_log=f'{cfg.outputs_dir}/tb/',
        device='cpu',
        seed=seed,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=cfg.train.checkpoint_freq,
            save_path=cfg.models_dir,
            name_prefix='ppo_shared',
        ),
        SyncingEvalCallback(
            eval_env,
            best_model_save_path=f'{cfg.models_dir}/best',
            log_path=f'{cfg.outputs_dir}/eval_logs',
            eval_freq=cfg.train.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=cfg.train.eval_episodes,
        ),
    ]

    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = cfg.model_path('ppo_shared_final')
    model.save(final_path)
    env.save(f'{cfg.models_dir}/vecnormalize.pkl')
    write_manifest(cfg, env, final_path)

    env.close()
    eval_env.close()
    print('Training complete.')
    print(f'TensorBoard: tensorboard --logdir {cfg.outputs_dir}/tb/')
    print(f'Best model : {cfg.models_dir}/best/best_model.zip')
    print(f'Final model: {final_path}.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO for TSC (standalone).')
    parser.add_argument('--sanity', action='store_true',
                        help='Run TLS sanity check and exit')
    args = parser.parse_args()

    # Recommended default when invoked standalone: offpeak + EMV so
    # VecNormalize sees non-zero EMV features during training.
    cfg = Config(scenario=ScenarioConfig(name="offpeak", emv=True))

    if args.sanity:
        sanity_check_tls(cfg)
    else:
        main(cfg)