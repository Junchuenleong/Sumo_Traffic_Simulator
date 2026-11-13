"""
Central configuration — single source of truth for every tunable in the pipeline.

Usage:
    from config import Config, PRESETS

    cfg = Config()                          # defaults (offpeak, no EMV)
    cfg = PRESETS["offpeak_emv"]            # named preset
    cfg = Config(scenario=ScenarioConfig(name="rushhour", emv=True))  # custom

    # In a run script:
    route_files = cfg.route_files()         # ['scenarios/rushhour.rou.xml', 'scenarios/emv.rou.xml']
    prefix      = cfg.out_prefix("hybrid")  # 'outputs/hybrid_rushhour_emv'
    net         = cfg.network.net_file      # 'networks/multi_agent_clean.net.xml'

Command-line overrides are applied in `run.py` before passing `cfg` to a runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Dict

# Re-export for scripts that do `from config import EmergencyPreemption`
from emergency_preemption import EmergencyPreemption  # noqa: F401


# ---------------------------------------------------------------------------
# Paths — RELATIVE to the repo root. No absolute paths anywhere.
# ---------------------------------------------------------------------------

REPO_ROOT        = Path(__file__).resolve().parent
NETWORKS_DIR     = REPO_ROOT / "networks"
SCENARIOS_DIR    = REPO_ROOT / "scenarios"
OUTPUTS_DIR      = REPO_ROOT / "outputs"
MODELS_DIR       = REPO_ROOT / "models"
FIGURES_DIR      = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# Sub-configs — one dataclass per concern.
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Which SUMO network to drive."""
    net_file: str = str(NETWORKS_DIR / "multi_agent_clean.net.xml")
    # For generate_scenarios.py --regen-tls:
    source_net_file: str = str(NETWORKS_DIR / "semenyih_source.net.xml")


@dataclass
class ScenarioConfig:
    """Traffic demand scenario."""
    name: str = "offpeak"          # offpeak | rushhour | gridlock
    emv: bool = False              # if True, also include emv.rou.xml
    scenarios_dir: str = str(SCENARIOS_DIR)

    def route_files(self) -> List[str]:
        paths = [f"{self.scenarios_dir}/{self.name}.rou.xml"]
        if self.emv:
            paths.append(f"{self.scenarios_dir}/emv.rou.xml")
        return paths


@dataclass
class SimConfig:
    """Simulator knobs shared across methods."""
    num_seconds: int = 3600
    delta_time: int = 5            # RL control tick, seconds
    yellow_time: int = 2           # sumo-rl yellow for RL / hybrid
    min_green: int = 5             # sumo-rl min-green
    max_green: int = 50            # sumo-rl max-green
    time_to_teleport: int = 120    # seconds stuck before SUMO teleports.
                                   # NOTE: your older run_fixed/maxpressure scripts
                                   # used 180. Unifying to 120 (matching PPO's value)
                                   # makes the controller comparison fairer.


@dataclass
class FixedTimeConfig:
    """Classical fixed-time + MaxPressure controllers."""
    min_green: int = 10
    yellow_time: int = 3


@dataclass
class PreemptionConfig:
    """Emergency-vehicle preemption (EmergencyPreemption.__init__)."""
    detection_distance: float = 250.0
    release_distance: float = 300.0        # hysteresis; must be >= detection
    eta_threshold_s: float = 25.0          # skip preempting for EMVs > 25s away
    always_preempt_within_m: float = 50.0  # ... unless EMV is practically on top of us
    detection_probability: float = 1.0     # 1.0 = perfect detector; lower to sim real hw
    comm_latency_steps: int = 0            # ticks of switch delay
    debug: bool = False


@dataclass
class RewardConfig:
    """Shaped reward coefficients (obs_reward_wrapper.ObsRewardWrapper).

    Defaults match what your latest train_ppo.py / train_ppo_hybrid.py /
    run_ppo_eval.py all use — so train, eval, and hybrid see the same shaping.
    """
    spillback_penalty_coef: float = 0.10
    emv_stop_penalty_coef: float = 2.00
    switch_penalty_coef: float = 0.05
    spillback_threshold: float = 0.80
    emv_detection_distance: float = 250.0


@dataclass
class TrainConfig:
    """PPO training hyper-parameters. Matches your latest train_ppo.py."""
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    seed: int = 42
    checkpoint_freq: int = 50_000
    eval_freq: int = 50_000
    eval_episodes: int = 1


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    network:    NetworkConfig    = field(default_factory=NetworkConfig)
    scenario:   ScenarioConfig   = field(default_factory=ScenarioConfig)
    sim:        SimConfig        = field(default_factory=SimConfig)
    fixed_time: FixedTimeConfig  = field(default_factory=FixedTimeConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    reward:     RewardConfig     = field(default_factory=RewardConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    outputs_dir: str = str(OUTPUTS_DIR)
    models_dir:  str = str(MODELS_DIR)

    def route_files(self) -> List[str]:
        return self.scenario.route_files()

    def route_files_str(self) -> str:
        """Comma-joined — SUMO accepts this via -r."""
        return ",".join(self.route_files())

    def out_prefix(self, controller: str) -> str:
        """Unique output prefix per (controller, scenario, EMV) combo so runs
        with different options don't overwrite each other's tripinfo."""
        tag = self.scenario.name
        if self.scenario.emv:
            tag += "_emv"
        return f"{self.outputs_dir}/{controller}_{tag}"

    def model_path(self, name: str = "ppo_shared_final") -> str:
        return f"{self.models_dir}/{name}"

    def with_(self, **overrides) -> "Config":
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Named presets — shortcuts for the combos you actually run often.
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Config] = {
    "offpeak":       Config(),
    "offpeak_emv":   Config(scenario=ScenarioConfig(name="offpeak",  emv=True)),
    "rushhour":      Config(scenario=ScenarioConfig(name="rushhour", emv=False)),
    "rushhour_emv":  Config(scenario=ScenarioConfig(name="rushhour", emv=True)),
    "gridlock":      Config(scenario=ScenarioConfig(name="gridlock", emv=False)),
    "gridlock_emv":  Config(scenario=ScenarioConfig(name="gridlock", emv=True)),
}


# ---------------------------------------------------------------------------
# Back-compat: old code imports `NET_FILE` from this module. Keep working.
# ---------------------------------------------------------------------------

NET_FILE = NetworkConfig().net_file