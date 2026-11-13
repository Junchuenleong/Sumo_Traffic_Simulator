"""
Central configuration — single source of truth for every tunable in the pipeline.

Usage:
    from config import Config, PRESETS

    cfg = Config()                          # tier from $TSC_TIER (default: "real")
    cfg = Config(tier="grid")               # explicit
    cfg = PRESETS["offpeak_emv"]            # named preset
    cfg = Config(scenario=ScenarioConfig(name="rushhour", emv=True))

    # In a run script:
    route_files = cfg.route_files()         # ['scenarios/<tier>/rushhour.rou.xml', ...]
    prefix      = cfg.out_prefix("hybrid")  # 'outputs/<tier>/hybrid_rushhour_emv'
    net         = cfg.network.net_file      # auto-selected from tier

Tier selection:
    Bats / shells set TSC_TIER=grid or TSC_TIER=real before invoking run.py;
    Config picks it up automatically. Override per-call via Config(tier=...).

Network selection:
    The .net.xml file is auto-resolved from the active tier via _TIER_NETS
    (see below). To use a different real-network file (e.g. for a different
    OSM city), edit the "real" entry in _TIER_NETS — that's the one place.
    To force a specific net regardless of tier, pass it explicitly:
        Config(network=NetworkConfig(net_file="networks/whatever.net.xml"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

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
# Tier registry
# ---------------------------------------------------------------------------

ALLOWED_TIERS = ("real", "grid")

# Per-tier network file. Edit the "real" entry to change which OSM net is
# used; edit the "grid" entry only if you renamed the synthetic-grid output.
_TIER_NETS: Dict[str, str] = {
    "grid": str(NETWORKS_DIR / "grid.net.xml"),
    "real": str(NETWORKS_DIR / "multi_agent_clean.net.xml"),
}


def _default_tier() -> str:
    """Read TSC_TIER from the environment. Anything outside ALLOWED_TIERS
    falls back to 'real' with a warning so a typo doesn't silently dump
    results into the wrong folder."""
    t = os.environ.get("TSC_TIER", "real").strip().lower()
    if t not in ALLOWED_TIERS:
        print(f"[config] WARNING: TSC_TIER={t!r} is not one of "
              f"{ALLOWED_TIERS}; falling back to 'real'.")
        t = "real"
    return t


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Which SUMO network to drive.

    `net_file` defaults to None so Config.__post_init__ can fill it from
    the active tier via _TIER_NETS. Set it explicitly to bypass that.
    """
    net_file: Optional[str] = None
    # For generate_scenarios.py --regen-tls (only meaningful for the OSM tier):
    source_net_file: str = str(NETWORKS_DIR / "semenyih_source.net.xml")


@dataclass
class ScenarioConfig:
    """Traffic demand scenario."""
    name: str = "offpeak"                # offpeak | rushhour | gridlock
    emv: bool = False                    # if True, also include emv.rou.xml
    scenarios_dir: str = str(SCENARIOS_DIR)  # overridden by Config.__post_init__

    def route_files(self) -> List[str]:
        paths = [f"{self.scenarios_dir}/{self.name}.rou.xml"]
        if self.emv:
            paths.append(f"{self.scenarios_dir}/emv.rou.xml")
        return paths


@dataclass
class SimConfig:
    num_seconds: int = 3600
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50
    time_to_teleport: int = 120


@dataclass
class FixedTimeConfig:
    min_green: int = 10
    yellow_time: int = 3


@dataclass
class PreemptionConfig:
    detection_distance: float = 250.0
    release_distance: float = 300.0
    eta_threshold_s: float = 25.0
    always_preempt_within_m: float = 50.0
    detection_probability: float = 1.0
    comm_latency_steps: int = 0
    debug: bool = False


@dataclass
class RewardConfig:
    spillback_penalty_coef: float = 0.10
    emv_stop_penalty_coef: float = 2.00
    switch_penalty_coef: float = 0.05
    spillback_threshold: float = 0.80
    emv_detection_distance: float = 250.0


@dataclass
class TrainConfig:
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

    # "real" or "grid". Default is read from $TSC_TIER at construction time.
    tier: str = field(default_factory=_default_tier)

    def __post_init__(self) -> None:
        if self.tier not in ALLOWED_TIERS:
            print(f"[config] WARNING: tier={self.tier!r} is not one of "
                  f"{ALLOWED_TIERS}; falling back to 'real'.")
            self.tier = "real"

        # Tier-aware network selection. If the user didn't explicitly set
        # network.net_file, pick the canonical net for this tier.
        if self.network.net_file is None:
            self.network.net_file = _TIER_NETS[self.tier]

        # Thread tier into scenario.scenarios_dir, but only if the user
        # didn't explicitly override it. Custom dirs are left alone.
        if self.scenario.scenarios_dir == str(SCENARIOS_DIR):
            self.scenario.scenarios_dir = str(SCENARIOS_DIR / self.tier)
            
    def banner(self) -> None:
        """Print a one-line summary of the active config. Call once, after the
        final cfg is fully built — not from __post_init__, which fires on every
        Config() and replace() call."""
        src = "TSC_TIER env" if "TSC_TIER" in os.environ else "default (TSC_TIER unset)"
        print(f"[config] tier={self.tier}  scenario={self.scenario.name}  emvs_in_routes={self.scenario.emv}  ({src})")

    def banner_tier(self) -> None:
        """Print a one-line summary of the active config. Call once, after the
        final cfg is fully built — not from __post_init__, which fires on every
        Config() and replace() call."""
        src = "TSC_TIER env" if "TSC_TIER" in os.environ else "default (TSC_TIER unset)"
        print(f"[config] tier={self.tier}  scenario={self.scenario.name} ({src})")

    # outputs_dir / models_dir are properties (not fields) so they always
    # reflect the current tier without re-init gymnastics.
    @property
    def outputs_dir(self) -> str:
        return str(OUTPUTS_DIR / self.tier)

    @property
    def models_dir(self) -> str:
        return str(MODELS_DIR / self.tier)

    def route_files(self) -> List[str]:
        return self.scenario.route_files()

    def route_files_str(self) -> str:
        return ",".join(self.route_files())

    def out_prefix(self, controller: str) -> str:
        tag = self.scenario.name
        if self.scenario.emv:
            tag += "_emv"
        return f"{self.outputs_dir}/{controller}_{tag}"

    def model_path(self, name: str = "ppo_shared_final") -> str:
        return f"{self.models_dir}/{name}"

    def with_(self, **overrides) -> "Config":
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Named presets
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
# Back-compat: old code imports `NET_FILE` from this module. Resolve from
# the env tier so legacy callers see the right net for the active run.
# ---------------------------------------------------------------------------

NET_FILE = _TIER_NETS[_default_tier()]