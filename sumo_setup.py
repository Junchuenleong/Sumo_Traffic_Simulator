"""
Sets up SUMO_HOME and adds SUMO's tools + bin directories to the Python path
and system PATH. Import this module before importing sumo_rl or traci.
"""

import os
import sys

SUMO_HOME = os.environ.get("SUMO_HOME")

if not SUMO_HOME:
    raise EnvironmentError(
        "SUMO_HOME is not set. Please set it to your SUMO installation folder "
        "(e.g. C:\\Program Files (x86)\\Eclipse\\Sumo on Windows)."
    )

# Make SUMO's Python tools importable (traci, sumolib, etc.)
tools = os.path.join(SUMO_HOME, "tools")
if tools not in sys.path:
    sys.path.append(tools)

# Make sure sumo / sumo-gui binaries are findable
bin_dir = os.path.join(SUMO_HOME, "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bin_dir

# 4. Define the exact tool paths your main script asks for
is_windows = sys.platform.startswith('win')

NETCONVERT = os.path.join(SUMO_HOME, "bin", f"netconvert{'.exe' if is_windows else ''}")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")