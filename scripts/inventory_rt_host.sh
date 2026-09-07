#!/usr/bin/env bash
# Read-only inventory of an FR3 real-time control host.
# This script does not connect to or command the robot and does not install anything.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-$REPO_ROOT/rt_host_inventory_${TIMESTAMP}.txt}"

# Record everything to the terminal and to a file that can be shared afterward.
exec > >(tee "$OUTPUT") 2>&1

section() {
    printf '\n===== %s =====\n' "$1"
}

run() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@" || true
}

section "Inventory metadata"
printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Output: %s\n' "$OUTPUT"

section "Operating system and kernel"
run uname -a
run uname -r
run cat /etc/os-release

section "Python executables"
run command -v python
run command -v python3
run command -v pip
run command -v pip3
run python --version
run python3 --version

section "Environment managers"
run command -v conda
run command -v mamba
run command -v micromamba
if command -v conda >/dev/null 2>&1; then
    run conda env list
fi

section "Existing Python environments under home"
printf '$ find %q -maxdepth 5 for bin/python and bin/python3\n' "$HOME"
find "$HOME" -maxdepth 5 -type f \
    \( -path '*/bin/python' -o -path '*/bin/python3' \) \
    2>/dev/null | sort || true

section "Franka-related apt packages"
printf '$ dpkg -l | grep -Ei franka/libfranka/franky\n'
dpkg -l 2>/dev/null | grep -Ei 'franka|libfranka|franky' || true

section "Franka libraries known to the linker"
printf '$ ldconfig -p | grep -Ei franka/franky\n'
ldconfig -p 2>/dev/null | grep -Ei 'franka|franky' || true

section "Franka pkg-config"
run pkg-config --modversion libfranka
run pkg-config --cflags --libs libfranka

section "Potential Franka installations (first 200 results)"
printf '$ find /usr /usr/local /opt %q for Franka-related paths\n' "$HOME"
find /usr /usr/local /opt "$HOME" -maxdepth 5 \
    \( -iname '*libfranka*' -o -iname '*franky*' -o -iname '*franka*control*' \) \
    2>/dev/null | sort | head -200 || true

section "Existing robot or controller processes"
printf '$ ps aux | grep robot-related names\n'
ps aux | grep -Ei 'franka|franky|fr3|libfranka|controller' | grep -v grep || true

section "Related systemd services"
printf '$ systemctl list-units --type=service --all | grep robot-related names\n'
systemctl list-units --type=service --all 2>/dev/null \
    | grep -Ei 'franka|franky|fr3|robot' || true

section "User and realtime permissions"
run id
run groups
printf '$ ulimit -r\n'
ulimit -r || true
printf '$ ulimit -l\n'
ulimit -l || true

section "Network interfaces"
run ip -br address

section "Network routes"
run ip route

section "Repository state"
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    run git -C "$REPO_ROOT" status --short
    run git -C "$REPO_ROOT" log -1 --oneline
    run git -C "$REPO_ROOT" remote -v
else
    printf 'Not inside a Git working tree.\n'
fi

section "Manual details still needed"
printf '%s\n' \
    'Please provide separately:' \
    '1. FR3 FCI IP shown in Desk.' \
    '2. FR3 system/firmware version shown in Desk.' \
    '3. Previous control stack (FR3Py, ROS, libfranka, or other).' \
    '4. Whether a robot controller starts automatically at boot.' \
    '5. Whether robot Ethernet and lab LAN use separate interfaces.'

section "Complete"
printf 'Inventory saved to: %s\n' "$OUTPUT"
printf 'This script made no system changes and issued no robot commands.\n'
