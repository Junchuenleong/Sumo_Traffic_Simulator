"""
Unified CLI entrypoint for the traffic-signal-control pipeline.

Examples:
    # Baselines (no EMV)
    python run.py fixed        --scenario offpeak
    python run.py maxpressure  --scenario offpeak

    # Baselines + emergency preemption
    python run.py fixed        --scenario offpeak --emv
    python run.py maxpressure  --scenario rushhour --emv

    # PPO
    python run.py train        --scenario offpeak --emv --timesteps 500000 --seed 42
    python run.py ppo          --scenario offpeak --emv
    python run.py hybrid       --scenario offpeak --emv

    # Housekeeping
    python run.py generate                                   # build .rou.xml scenarios
    python run.py evaluate     --scenario offpeak            # compare all controllers

Tier:
    Set TSC_TIER=grid or TSC_TIER=real before invoking. All paths
    (scenarios, outputs, models, default network file) follow the tier
    automatically. Default if unset: real.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import sumo_setup  # noqa: F401

from config import Config, PRESETS, ScenarioConfig


CONTROLLERS = ["fixed", "maxpressure", "ppo", "hybrid"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _common(subp):
        subp.add_argument("--scenario", default="offpeak",
                          choices=["offpeak", "rushhour", "gridlock"])
        subp.add_argument("--emv", action="store_true",
                          help="Include emv.rou.xml in the simulation")
        subp.add_argument("--preempt", action="store_true",
                          help="Enable EMV preemption in the controller "
                               "(requires --emv to do anything useful)")
        subp.add_argument("--gui", action="store_true",
                          help="Open SUMO-GUI (slower)")
        subp.add_argument("--seed", type=int, default=0)
        subp.add_argument("--preset", default=None,
                          help=f"Use a named preset (one of: {list(PRESETS)}) "
                               f"as the starting config")
        subp.add_argument("--detection-distance", type=float, default=None,
                          help="Override preemption detection distance (meters)")
        subp.add_argument("--eta-threshold", type=float, default=None,
                          help="Override preemption ETA gate (seconds)")

    for ctrl in CONTROLLERS:
        sp = sub.add_parser(ctrl, help=f"Run the {ctrl} controller")
        _common(sp)
        if ctrl in ("ppo", "hybrid"):
            # Default None -> resolve via cfg.model_path() at dispatch time so
            # the active tier (TSC_TIER=grid|real) picks the right folder.
            sp.add_argument("--model", default=None,
                            help="Path to trained PPO model (.zip omitted). "
                                 "Defaults to <models_dir>/ppo_shared_final, "
                                 "where models_dir follows TSC_TIER.")
        if ctrl == "hybrid":
            sp.add_argument("--inspect-emv", action="store_true",
                            help="Print EMV-route vs TLS-coverage diagnostic")
            sp.add_argument("--debug-preempt", action="store_true",
                            help="Verbose preemption logging")

    # train
    sp_tr = sub.add_parser("train", help="Train PPO")
    _common(sp_tr)
    sp_tr.add_argument("--timesteps", type=int, default=500_000)

    # generate
    sp_gen = sub.add_parser("generate",
                            help="(Re)build the scenario .rou.xml files")
    sp_gen.add_argument("--regen-tls", action="store_true",
                        help="Rebuild TLS programs from the source net "
                             "(DESTRUCTIVE — wipes manual TLS edits)")

    # evaluate — only needs --scenario (no EMV flag since it compares both)
    sp_ev = sub.add_parser("evaluate", help="Print comparison table from outputs/")
    sp_ev.add_argument("--scenario", default="offpeak",
                       choices=["offpeak", "rushhour", "gridlock"],
                       help="Which scenario's results to compare")

    return p


def cfg_from_args(args) -> Config:
    """Build a Config from CLI args, starting from --preset or defaults."""
    if getattr(args, "preset", None) is not None:
        if args.preset not in PRESETS:
            sys.exit(f"Unknown preset '{args.preset}'. "
                     f"Available: {sorted(PRESETS)}")
        cfg = PRESETS[args.preset]
    else:
        cfg = Config()

    cfg = replace(
        cfg,
        scenario=ScenarioConfig(name=args.scenario, emv=bool(getattr(args, "emv", False))),
    )

    # Preemption overrides
    preempt_overrides = {}
    if getattr(args, "detection_distance", None) is not None:
        preempt_overrides["detection_distance"] = args.detection_distance
        preempt_overrides["release_distance"] = max(
            cfg.preemption.release_distance,
            args.detection_distance * 1.2,
        )
    if getattr(args, "eta_threshold", None) is not None:
        preempt_overrides["eta_threshold_s"] = args.eta_threshold
    if getattr(args, "debug_preempt", False):
        preempt_overrides["debug"] = True
    if preempt_overrides:
        cfg = replace(cfg, preemption=replace(cfg.preemption, **preempt_overrides))

    # Train overrides
    train_overrides = {}
    if getattr(args, "timesteps", None) is not None:
        train_overrides["total_timesteps"] = args.timesteps
    if getattr(args, "seed", None) is not None:
        train_overrides["seed"] = args.seed
    if train_overrides:
        cfg = replace(cfg, train=replace(cfg.train, **train_overrides))

    os.makedirs(cfg.outputs_dir, exist_ok=True)
    os.makedirs(cfg.models_dir,  exist_ok=True)

    cfg.banner()
    return cfg


def dispatch(args) -> None:
    if args.command == "generate":
        # Real-network generator only. Grid scenarios use generate_grid_scenario.py.
        tier_env = os.environ.get("TSC_TIER", "real").strip().lower()
        if tier_env == "grid":
            sys.exit("[run] 'generate' is for the real network. "
                     "For grid, run: python generate_grid_scenario.py --scenario <name>")
        import generate_scenarios
        generate_scenarios.main(regen_tls=args.regen_tls, tier="real")
        return

    if args.command == "evaluate":
        import evaluate
        evaluate.main(scenario=args.scenario)
        return

    cfg = cfg_from_args(args)

    # Resolve a default --model path AFTER cfg is built, so models_dir
    # reflects the active tier.
    model_path = getattr(args, "model", None)
    if model_path is None:
        model_path = cfg.model_path("ppo_shared_final")

    if args.command == "fixed":
        if getattr(args, "preempt", False) and not cfg.scenario.emv:
            print("[run] WARNING: --preempt without --emv — no ambulances to preempt for.")
        if getattr(args, "preempt", False):
            import run_fixed_emv as mod
        else:
            import run_fixed as mod
        mod.run(cfg, gui=args.gui)

    elif args.command == "maxpressure":
        if getattr(args, "preempt", False) and not cfg.scenario.emv:
            print("[run] WARNING: --preempt without --emv — no ambulances to preempt for.")
        if getattr(args, "preempt", False):
            import run_maxpressure_emv as mod
        else:
            import run_maxpressure as mod
        mod.run(cfg, gui=args.gui)

    elif args.command == "ppo":
        import run_ppo_eval
        run_ppo_eval.evaluate(
            cfg,
            model_path=model_path,
            gui=args.gui,
            seed=args.seed,
        )

    elif args.command == "hybrid":
        import train_ppo_hybrid
        train_ppo_hybrid.run_hybrid(
            cfg,
            model_path=model_path,
            gui=args.gui,
            seed=args.seed,
            debug_preempt=args.debug_preempt,
            inspect_emv=args.inspect_emv,
        )

    elif args.command == "train":
        import train_ppo
        train_ppo.main(cfg)

    else:
        sys.exit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    dispatch(args)