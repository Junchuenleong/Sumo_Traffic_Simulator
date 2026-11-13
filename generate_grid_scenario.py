"""
Generate a synthetic grid scenario for TSC training / evaluation.

Pipeline-aligned: output paths follow config.py's directory layout AND its
tier convention (scenarios/<tier>/...). Default tier is "grid" — i.e.
running this script with no flags writes to scenarios/grid/, which is what
Config(tier="grid") will look up later.

Outputs (defaults, for --tier grid):
    networks/grid.net.xml                       - N x N grid with TLS at every intersection
    scenarios/grid/<scenario>.rou.xml           - randomized background traffic
    scenarios/grid/emv.rou.xml                  - sparse emergency-vehicle trips
    scenarios/grid/emv_types.add.xml            - emergency vType (referenced by emv.rou.xml)

Usage:
    python generate_grid_scenario.py                              # offpeak, default sizes
    python generate_grid_scenario.py --scenario rushhour
    python generate_grid_scenario.py --size 5 --scenario gridlock
    python generate_grid_scenario.py --tier grid_v2               # alt tier folder
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import sumo_setup  # noqa: F401  sets SUMO_HOME / PATH

from config import (
    NETWORKS_DIR,
    SCENARIOS_DIR,
    NetworkConfig,
    ScenarioConfig,
    SimConfig,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NET_FILE = str(NETWORKS_DIR / "grid.net.xml")
DEFAULT_TIER     = "grid"

GRID_SIZE    = 4          # 4x4 intersections
BLOCK_LENGTH = 200        # meters between adjacent intersections
LANE_COUNT   = 2
SPEED        = 13.89      # m/s (~50 km/h)

SIM_END    = SimConfig().num_seconds
EMV_PERIOD = 600.0

DEFAULT_SCENARIO = ScenarioConfig().name

# Names MUST match ScenarioConfig.name values (offpeak | rushhour | gridlock)
SCENARIOS = {
    'offpeak':  {'period': 2.5, 'fringe': 2,  'seed': 1},
    'rushhour': {'period': 0.8, 'fringe': 10, 'seed': 2},
    'gridlock': {'period': 0.3, 'fringe': 20, 'seed': 3},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sumo_tools() -> Path:
    home = os.environ.get('SUMO_HOME')
    if not home:
        sys.exit('ERROR: SUMO_HOME is not set. Install SUMO and export SUMO_HOME.')
    tools = Path(home) / 'tools'
    if not (tools / 'randomTrips.py').is_file():
        sys.exit(f'ERROR: randomTrips.py not found under {tools}.')
    return tools


def _run(cmd, desc: str) -> None:
    print(f'\n>>> {desc}')
    print(f'    {" ".join(str(c) for c in cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f'Command failed: {desc}')


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def tier_dir(tier: str) -> Path:
    return SCENARIOS_DIR / tier


def route_file_for(tier: str, scenario: str) -> str:
    """Path that ScenarioConfig(name=scenario).route_files() expects when
    Config.tier == tier."""
    return str(tier_dir(tier) / f'{scenario}.rou.xml')


def emv_route_path(tier: str) -> str:
    return str(tier_dir(tier) / 'emv.rou.xml')


def emv_types_path(tier: str) -> str:
    return str(tier_dir(tier) / 'emv_types.add.xml')


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def generate_net(size: int, block_length: int, lanes: int,
                 speed: float, out: str) -> None:
    _ensure_parent(out)
    cmd = [
        'netgenerate',
        '--grid',
        f'--grid.number={size}',
        f'--grid.length={block_length}',
        f'--default.lanenumber={lanes}',
        f'--default.speed={speed}',
        '--tls.guess=true',
        '--tls.default-type=static',
        '--no-turnarounds',
        '--output-file', out,
    ]
    _run(cmd, f'Generating {size}x{size} grid network -> {out}')


def generate_background_traffic(net: str, out: str, end: int,
                                period: float, seed: int, fringe: int = 5) -> None:
    _ensure_parent(out)
    tools = _sumo_tools()
    cmd = [
        sys.executable, str(tools / 'randomTrips.py'),
        '-n', net,
        '-r', out,
        '-e', str(end),
        '-p', str(period),
        '--fringe-factor', str(fringe),
        '--validate',
        '--seed', str(seed),
    ]
    _run(cmd, f'Generating background traffic -> {out}')


def write_emv_types(path: str) -> None:
    _ensure_parent(path)
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<additional>\n'
        '    <vType id="emergency" vClass="emergency" color="1,0,0"\n'
        '           accel="2.6" decel="4.5" maxSpeed="25"\n'
        '           guiShape="emergency" speedFactor="1.3"/>\n'
        '</additional>\n'
    )
    with open(path, 'w') as f:
        f.write(content)
    print(f'Wrote EMV vType file -> {path}')


def generate_emvs(net: str, out: str, types_file: str,
                  end: int, period: float, seed: int) -> None:
    """Sparse emergency vehicles, fringe-to-fringe so they cross multiple TLSs."""
    _ensure_parent(out)
    write_emv_types(types_file)
    tools = _sumo_tools()
    begin = 300
    cmd = [
        sys.executable, str(tools / 'randomTrips.py'),
        '-n', net,
        '-r', out,
        '-b', str(begin),
        '-e', str(end - begin),
        '-p', str(period),
        '--fringe-factor', '100',
        '--additional-file', types_file,
        '--trip-attributes', 'type="emergency" departSpeed="max"',
        '--prefix', 'emv_',
        '--validate',
        '--seed', str(seed),
    ]
    _run(cmd, f'Generating emergency-vehicle trips -> {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--size', type=int, default=GRID_SIZE,
                        help=f'Grid side length in intersections (default: {GRID_SIZE})')
    parser.add_argument('--block', type=int, default=BLOCK_LENGTH,
                        help=f'Block length in metres (default: {BLOCK_LENGTH})')
    parser.add_argument('--lanes', type=int, default=LANE_COUNT,
                        help=f'Lanes per direction (default: {LANE_COUNT})')
    parser.add_argument('--speed', type=float, default=SPEED,
                        help=f'Edge speed limit in m/s (default: {SPEED})')
    parser.add_argument('--end', type=int, default=SIM_END,
                        help=f'Simulation horizon in seconds (default: {SIM_END})')
    parser.add_argument('--scenario', choices=list(SCENARIOS.keys()),
                        default=DEFAULT_SCENARIO,
                        help=f'Traffic scenario (default: {DEFAULT_SCENARIO})')
    parser.add_argument('--emv-period', type=float, default=EMV_PERIOD,
                        help=f'Avg seconds between EMV insertions (default: {EMV_PERIOD})')
    parser.add_argument('--tier', default=DEFAULT_TIER,
                        help=f'Scenarios sub-folder under scenarios/. '
                             f'Must match Config.tier at runtime '
                             f'(default: {DEFAULT_TIER})')
    parser.add_argument('--net-out', default=DEFAULT_NET_FILE,
                        help=f'Net output path (default: {DEFAULT_NET_FILE})')
    parser.add_argument('--rou-out', default=None,
                        help='Route output path (default: scenarios/<tier>/<scenario>.rou.xml)')
    parser.add_argument('--emv-out', default=None,
                        help='EMV route output path (default: scenarios/<tier>/emv.rou.xml)')
    args = parser.parse_args()

    scenario_params = SCENARIOS[args.scenario]
    period = scenario_params['period']
    fringe = scenario_params['fringe']
    seed = scenario_params['seed']

    # Tier-aware default paths so the generated files line up with what
    # Config(tier=args.tier) will look up later.
    if args.rou_out is None:
        args.rou_out = route_file_for(args.tier, args.scenario)
    if args.emv_out is None:
        args.emv_out = emv_route_path(args.tier)
    types_file = emv_types_path(args.tier)

    generate_net(args.size, args.block, args.lanes, args.speed, args.net_out)
    generate_background_traffic(args.net_out, args.rou_out, args.end,
                                period, seed, fringe)
    generate_emvs(args.net_out, args.emv_out, types_file, args.end,
                  args.emv_period, seed + 1)

    default_net = NetworkConfig().net_file

    print('\n' + '=' * 60)
    print(f'Grid scenario ({args.scenario}) ready under tier "{args.tier}".')
    print('=' * 60)
    print(f'  Net file   : {args.net_out}')
    print(f'  Routes     : {args.rou_out}')
    print(f'  EMV routes : {args.emv_out}')
    print(f'  vType file : {types_file}')
    print(f'  Period     : {period}s between vehicles')
    print(f'  Fringe     : {fringe}')
    print(f'  Seed       : {seed}')

    if default_net is None or os.path.abspath(args.net_out) != os.path.abspath(default_net):
        try:
            rel = Path(args.net_out).resolve().relative_to(Path.cwd().resolve())
            rel_posix = rel.as_posix()
        except ValueError:
            rel_posix = Path(args.net_out).as_posix()

        print('\nNote: NetworkConfig.net_file currently defaults to')
        print(f'  {default_net}')
        print('which is NOT the file you just generated. To make the rest of')
        print('the pipeline use this grid net, either:')
        print(f'  - edit config.py:')
        print(f'      net_file: str = str(REPO_ROOT / "{rel_posix}")')
        print(f'    (or, if simpler: str(NETWORKS_DIR / "{Path(args.net_out).name}"))')
        print(f'  - re-run with --net-out "{default_net}" to overwrite the default target.')

    print('\nNext steps (make sure TSC_TIER matches the --tier above):')
    print(f'  set TSC_TIER={args.tier}')
    print(f'  python run.py train       --scenario {args.scenario} --emv')
    print(f'  python run.py fixed       --scenario {args.scenario}')
    print(f'  python run.py maxpressure --scenario {args.scenario} --emv')
    print(f'  python run.py evaluate    --scenario {args.scenario}')


if __name__ == '__main__':
    main()
