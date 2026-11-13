"""
Shared helper for assembling SUMO additional-command strings so train/eval/hybrid
all use the same flags. Prevents silent drift between scripts.
"""


def build_sumo_cmd(
    out_prefix: str | None = None,
    teleport: int = 120,
    log_level: str = "quiet",   # 'quiet' | 'normal' | 'verbose'
    extra: str = "",
) -> str:
    """
    Build the string passed to sumo_rl's `additional_sumo_cmd` argument.

    Parameters
    ----------
    out_prefix : Optional path prefix. If provided, tripinfo + emission
        outputs are written to `<out_prefix>_tripinfo.xml` and
        `<out_prefix>_emissions.xml`.
    teleport : Seconds before SUMO teleports a stuck vehicle. 120 is
        aggressive enough to break deadlocks but lenient enough to let
        legitimate queueing happen.
    log_level : 'quiet' suppresses step/warning spam. Use 'verbose' only
        while debugging a specific issue.
    extra : Arbitrary extra flags appended verbatim.
    """
    parts = []

    if out_prefix:
        parts.append(f'--tripinfo-output {out_prefix}_tripinfo.xml')
        parts.append(f'--emission-output {out_prefix}_emissions.xml')

    # Teleport + routing robustness. Without these, one bad merge freezes
    # the whole episode and your tripinfo metrics become gridlock noise.
    parts.append('--ignore-route-errors true')
    parts.append('--collision.action warn')   # don't abort on physics collisions

    if log_level == 'quiet':
        parts.append('--no-step-log true')
        parts.append('--no-warnings true')
    elif log_level == 'verbose':
        parts.append('--verbose true')
        parts.append('--no-step-log false')

    if extra:
        parts.append(extra)

    return ' ' + ' '.join(parts) + ' '


def parse_tripinfo(tripinfo_path: str) -> dict:
    """Read SUMO's tripinfo XML and return summary statistics. Use after
    every eval to check if teleports are dominating the results."""
    import xml.etree.ElementTree as ET
    import os

    if not os.path.exists(tripinfo_path):
        return {'error': f'file not found: {tripinfo_path}'}

    try:
        tree = ET.parse(tripinfo_path)
    except ET.ParseError as e:
        return {'error': f'parse failed: {e}'}

    root = tree.getroot()
    trips = root.findall('tripinfo')
    if not trips:
        return {'completed': 0, 'error': 'no tripinfos — SUMO aborted early?'}

    n_completed = len(trips)
    total_teleports = sum(int(t.get('teleports', '0') or '0') for t in trips)
    waiting = [float(t.get('waitingTime', '0') or '0') for t in trips]
    time_loss = [float(t.get('timeLoss', '0') or '0') for t in trips]
    duration = [float(t.get('duration', '0') or '0') for t in trips]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Emergency vehicles separately
    emv_trips = [t for t in trips if t.get('vType', '') == 'emergency']
    emv_duration = [float(t.get('duration', '0') or '0') for t in emv_trips]
    emv_waiting = [float(t.get('waitingTime', '0') or '0') for t in emv_trips]

    return {
        'completed': n_completed,
        'total_teleports': total_teleports,
        'teleport_rate_pct': 100.0 * total_teleports / max(n_completed, 1),
        'mean_waiting_s': mean(waiting),
        'mean_time_loss_s': mean(time_loss),
        'mean_duration_s': mean(duration),
        'n_emv_completed': len(emv_trips),
        'mean_emv_duration_s': mean(emv_duration),
        'mean_emv_waiting_s': mean(emv_waiting),
    }


def print_tripinfo_summary(tripinfo_path: str, label: str = '') -> None:
    """Pretty-print tripinfo stats — call at the end of every eval script."""
    stats = parse_tripinfo(tripinfo_path)
    tag = f'[{label}] ' if label else ''
    if 'error' in stats:
        print(f'{tag}tripinfo error: {stats["error"]}')
        return

    print(f'\n{tag}Tripinfo summary ({tripinfo_path})')
    print(f'  Completed trips     : {stats["completed"]}')
    print(f'  Total teleports     : {stats["total_teleports"]}'
          f'  ({stats["teleport_rate_pct"]:.1f}% of trips)')
    print(f'  Mean waiting time   : {stats["mean_waiting_s"]:.1f}s')
    print(f'  Mean time loss      : {stats["mean_time_loss_s"]:.1f}s')
    print(f'  Mean duration       : {stats["mean_duration_s"]:.1f}s')
    if stats['n_emv_completed'] > 0:
        print(f'  EMV completed       : {stats["n_emv_completed"]}')
        print(f'  Mean EMV duration   : {stats["mean_emv_duration_s"]:.1f}s')
        print(f'  Mean EMV waiting    : {stats["mean_emv_waiting_s"]:.1f}s')
    else:
        print(f'  EMV completed       : 0  (check route/depart times)')

    # Warn if gridlock is skewing the metrics
    if stats['teleport_rate_pct'] > 5.0:
        print(f'  ⚠ WARNING: {stats["teleport_rate_pct"]:.1f}% teleport rate — '
              f'your metrics reflect gridlock, not controller quality.')
