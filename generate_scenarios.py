"""
Generate traffic scenarios for the SUMO network.

MODES:
  --regen-tls     Rebuild TLS programs from the source net (DESTRUCTIVE:
                  wipes manual edits). Off by default.
  (default)       Just generate the three scenario route files and the
                  EMV route file. Assumes NetworkConfig.net_file already
                  exists and is how you want it.

Outputs (defaults, --tier real):
    scenarios/real/offpeak.rou.xml
    scenarios/real/rushhour.rou.xml
    scenarios/real/gridlock.rou.xml
    scenarios/real/emv.rou.xml

Usage:
    python generate_scenarios.py                 # routes only, safe to re-run
    python generate_scenarios.py --regen-tls     # rebuilds TLS programs!
    python generate_scenarios.py --tier real_v2  # alternate tier folder
"""

import argparse
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

import sumo_setup      # sets SUMO_HOME, PATH, tools dir on sys.path
import sumolib

from config import NetworkConfig, SCENARIOS_DIR, Config


# ---------------------------------------------------------------------------
# Paths — pulled from NetworkConfig so changing config.py changes everything.
# ---------------------------------------------------------------------------

CFG = Config()  # this triggers __post_init__
NET_FILE = CFG.network.net_file              # destination of --regen-tls AND
                                            # the net we read for route generation
SOURCE_NET = CFG.network.source_net_file       # raw OSMWebWizard output (for --regen-tls)


DEFAULT_TIER = "real"


# ---------------------------------------------------------------------------
# Tier-aware path helpers (mirror generate_grid_scenario.py)
# ---------------------------------------------------------------------------

def tier_dir(tier: str) -> Path:
    return SCENARIOS_DIR / tier


def route_file_for(tier: str, scenario: str) -> str:
    """Path that ScenarioConfig(name=scenario).route_files() expects when
    Config.tier == tier."""
    return str(tier_dir(tier) / f'{scenario}.rou.xml')


def emv_route_path(tier: str) -> str:
    return str(tier_dir(tier) / 'emv.rou.xml')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Optional TLS regeneration (DANGEROUS — wipes manual edits)
# ---------------------------------------------------------------------------

def regenerate_tls(source: str, dest: str) -> None:
    """Rebuild TLS programs from `source`, writing to `dest`.

    WARNING: If you have hand-tweaked TLS phases in the source net, this
    wipes them. Only run when you've just (re)built from OSM and want
    clean programs.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f'Source net not found: {source}')

    Path(dest).parent.mkdir(parents=True, exist_ok=True)

    orig = sumolib.net.readNet(source)
    tls_ids = [t.getID() for t in orig.getTrafficLights()]
    print(f'Preserving {len(tls_ids)} TLS: {tls_ids}')

    subprocess.run([
        sumo_setup.NETCONVERT,
        '--sumo-net-file', source,
        '-o', dest,
        '--tls.discard-loaded',
        '--tls.set', ','.join(tls_ids),
        '--tls.default-type', 'static',
        '--no-turnarounds',
        '--tls.layout', 'incoming',
        '--keep-edges.components', '1',
        '--remove-edges.isolated',
    ], check=True)
    print(f'[OK] TLS-regenerated net written to {dest}')


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------

def _is_source_fringe(e):
    return not [x for x in e.getFromNode().getIncoming()
                if not x.getID().startswith(':')]


def _is_sink_fringe(e):
    return not [x for x in e.getToNode().getOutgoing()
                if not x.getID().startswith(':')]


# Scenario intensities. `period` is the average seconds between spawns
# (smaller = denser traffic). `fringe` biases trips toward network
# boundaries (realistic for a city sub-area).
SCENARIOS = {
    'offpeak':  {'period': '2.5', 'fringe': '2',  'seed': '1'},
    'rushhour': {'period': '0.8', 'fringe': '10', 'seed': '2'},
    'gridlock': {'period': '0.3', 'fringe': '20', 'seed': '3'},
}


def generate_scenario_routes(tier: str) -> None:
    _ensure_dir(tier_dir(tier))
    for name, params in SCENARIOS.items():
        out = route_file_for(tier, name)
        print(f"Generating '{name}' traffic -> {out}")
        subprocess.run([
            sys.executable, sumo_setup.RANDOM_TRIPS,
            '-n', NET_FILE,
            '-r', out,
            '-e', '3600',
            '--period', params['period'],
            '--fringe-factor', params['fringe'],
            '--min-distance', '50',
            '--seed', params['seed'],
            '--validate',
            '--trip-attributes', 'departLane="best" departSpeed="max"',
        ], check=True)


# ---------------------------------------------------------------------------
# EMV route generation
# ---------------------------------------------------------------------------

# An EMV route should be long enough that preemption has time to fire
# before the EMV reaches the first TLS. At ~15 m/s typical approach and
# ~150-250m detection window, 400m gives at least ~25s of route to work
# with — enough for detection -> phase switch -> green.
MIN_EMV_ROUTE_LENGTH_M = 400.0
# Minimum TLS crossings on the route. 1 is the floor; the preferred-pair
# heuristic tries to maximise this implicitly by picking endpoints at
# opposite fringes.
MIN_EMV_ROUTE_TLS_HITS = 1


def node_xy(edge):
    return edge.getFromNode().getCoord()


def stable_seed(key: str) -> int:
    """Deterministic hash that doesn't depend on Python's randomised
    hash() — so route generation is reproducible across runs."""
    return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % (2**32)


def _route_length_m(route_edges) -> float:
    return float(sum(e.getLength() for e in route_edges))


def _tls_hits_on_route(route_edges, tls_edges_set) -> int:
    return sum(1 for e in route_edges if e.getID() in tls_edges_set)


def _tls_edge_set(net) -> set:
    tls_edges = set()
    for tls_obj in net.getTrafficLights():
        for e in tls_obj.getEdges():
            if not e.getID().startswith(':'):
                tls_edges.add(e.getID())
    return tls_edges


def find_routable_pair(net, source_pool, sink_pool,
                       prefer_from=None, prefer_to=None,
                       seed=None, max_fallback=300,
                       min_route_m: float = MIN_EMV_ROUTE_LENGTH_M,
                       min_tls_hits: int = MIN_EMV_ROUTE_TLS_HITS):
    """Pick a (source_edge, sink_edge, route) triple where:
      * source is emergency-routable to sink,
      * the route crosses >= min_tls_hits TLS-controlled edges,
      * the route is >= min_route_m long.

    source_pool and sink_pool should be *fringe* pools so the fallback
    stays realistic when the preferred pair doesn't fit.
    """
    rng = random.Random(seed)
    tls_edges = _tls_edge_set(net)
    if not tls_edges:
        print('  [warn] network has no TLS-controlled edges; '
              'dropping TLS-hit requirement.')
        min_tls_hits = 0

    def evaluate(from_e, to_e):
        if from_e.getID() == to_e.getID():
            return None
        route, _ = net.getShortestPath(from_e, to_e, vClass='emergency')
        if route is None:
            return None
        if _route_length_m(route) < min_route_m:
            return None
        if _tls_hits_on_route(route, tls_edges) < min_tls_hits:
            return None
        return route

    # 1) Try the caller's preferred pair first (e.g. west fringe -> east fringe).
    if prefer_from is not None and prefer_to is not None:
        route = evaluate(prefer_from, prefer_to)
        if route is not None:
            return prefer_from, prefer_to, route
        print(f'  [warn] Preferred pair {prefer_from.getID()} -> '
              f'{prefer_to.getID()} rejected '
              f'(too short / no TLS / unroutable); searching alternatives...')

    # 2) Fallback: random source-fringe -> random sink-fringe, until a
    #    pair satisfies the constraints.
    for attempt in range(max_fallback):
        f = rng.choice(source_pool)
        t = rng.choice(sink_pool)
        route = evaluate(f, t)
        if route is not None:
            print(f'  [ok] Fringe EMV pair found after {attempt + 1} attempts: '
                  f'{f.getID()} -> {t.getID()}  '
                  f'(route {_route_length_m(route):.0f}m, '
                  f'{_tls_hits_on_route(route, tls_edges)} TLS)')
            return f, t, route

    raise RuntimeError(
        f'Could not find a valid EMV pair after {max_fallback} attempts '
        f'(min_route_m={min_route_m}, min_tls_hits={min_tls_hits}). '
        'Check that your network has fringe-to-fringe emergency-class '
        'connectivity through at least one TLS.'
    )


def generate_emv_routes(tier: str) -> None:
    _ensure_dir(tier_dir(tier))
    out_path = emv_route_path(tier)
    print(f'Generating emergency vehicle scenario -> {out_path}')
    net = sumolib.net.readNet(NET_FILE)
    edges = [e for e in net.getEdges() if not e.getID().startswith(':')]

    candidate_edges = [e for e in edges
                       if e.allows('emergency') and e.getLength() > 30]

    source_fringe = [e for e in candidate_edges if _is_source_fringe(e)]
    sink_fringe   = [e for e in candidate_edges if _is_sink_fringe(e)]
    print(f'  {len(candidate_edges)} emergency-accessible edges '
          f'({len(source_fringe)} source-fringe, {len(sink_fringe)} sink-fringe)')

    if len(source_fringe) < 2 or len(sink_fringe) < 2:
        print('  [warn] Very few fringe edges; falling back to full candidate '
              'pool (EMV origins may look synthetic).')
        source_fringe = candidate_edges
        sink_fringe = candidate_edges

    west_src  = min(source_fringe, key=lambda e: node_xy(e)[0])
    east_snk  = max(sink_fringe,   key=lambda e: node_xy(e)[0])
    east_src  = max(source_fringe, key=lambda e: node_xy(e)[0])
    west_snk  = min(sink_fringe,   key=lambda e: node_xy(e)[0])
    south_src = min(source_fringe, key=lambda e: node_xy(e)[1])
    north_snk = max(sink_fringe,   key=lambda e: node_xy(e)[1])

    desired = [
        ('amb_1',  600, west_src,  east_snk),
        ('amb_2', 1500, south_src, north_snk),
        ('amb_3', 2400, east_src,  west_snk),
    ]

    ev_trips = []
    for vid, depart, pref_from, pref_to in desired:
        from_e, to_e, _route = find_routable_pair(
            net, source_fringe, sink_fringe,
            prefer_from=pref_from, prefer_to=pref_to,
            seed=stable_seed(vid),
        )
        ev_trips.append((vid, depart, from_e.getID(), to_e.getID()))

    with open(out_path, 'w') as f:
        f.write('<routes>\n')

        # EMV vType upgraded with junction-model params so the ambulance
        # behaves realistically when preemption fails OR hasn't fired yet.
        f.write(
            '    <vType id="emergency" vClass="emergency" color="1,0,0" '
            'guiShape="emergency" '
            'speedFactor="1.3" impatience="1.0" sigma="0.0" tau="1.2" '
            'accel="3.5" decel="5.0" emergencyDecel="9.0" '
            'jmDriveAfterRedTime="1" jmDriveRedSpeed="13.89" '
            'jmIgnoreFoeProb="0.3"/>\n'
        )
        for vid, depart, from_e, to_e in ev_trips:
            f.write(f'    <trip id="{vid}" type="emergency" depart="{depart}" '
                    f'from="{from_e}" to="{to_e}" '
                    f'departSpeed="avg" departLane="best"/>\n')
        f.write('</routes>\n')
    print(f'  [OK] Wrote {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(regen_tls: bool, tier: str) -> None:
    if regen_tls:
        print('⚠ --regen-tls is set: this will wipe any manual edits to TLS programs.')
        regenerate_tls(SOURCE_NET, NET_FILE)

    net = sumolib.net.readNet(NET_FILE)
    edges = [e for e in net.getEdges() if not e.getID().startswith(':')]
    tls_ids = [t.getID() for t in net.getTrafficLights()]
    print(f'{len(edges)} edges | {len(net.getNodes())} nodes | '
          f'{len(tls_ids)} traffic lights')

    generate_scenario_routes(tier)
    generate_emv_routes(tier)

    print(f'\nAll done. Files ready (tier="{tier}"):')
    print(f'  {NET_FILE}')
    for name in SCENARIOS:
        print(f'  {route_file_for(tier, name)}')
    print(f'  {emv_route_path(tier)}')
    print(f'\nMake sure TSC_TIER={tier} is set when invoking run.py / the bats.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-tls', action='store_true',
                        help='Rebuild TLS programs from NetworkConfig.source_net_file '
                             '(DESTRUCTIVE: wipes manual edits)')
    parser.add_argument('--tier', default=DEFAULT_TIER,
                        help=f'Scenarios sub-folder under scenarios/. '
                             f'Must match Config.tier at runtime '
                             f'(default: {DEFAULT_TIER})')
    args = parser.parse_args()
    main(regen_tls=args.regen_tls, tier=args.tier)