#!/usr/bin/env bash
# Read-only follow-up inventory for the FR3DataCollection/FR3Py deployment.
# It does not activate FCI, open a libfranka connection, or command the robot.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${1:-192.168.123.250}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${2:-$REPO_ROOT/rt_fr3_stack_${TIMESTAMP}.txt}"

exec > >(tee "$OUTPUT") 2>&1

section() { printf '\n===== %s =====\n' "$1"; }
run() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@" || true
}

section "Metadata"
printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'Robot IP candidate: %s\n' "$ROBOT_IP"
printf 'Output: %s\n' "$OUTPUT"

section "Route selected for robot IP"
run ip route get "$ROBOT_IP"

section "Neighbor table before probe"
run ip neigh show "$ROBOT_IP"

section "ICMP reachability (network only; no FCI connection)"
run ping -c 3 -W 1 "$ROBOT_IP"

section "Neighbor table after probe"
run ip neigh show "$ROBOT_IP"

section "Desk HTTPS reachability (headers only)"
run curl -k -sS -I --connect-timeout 3 --max-time 5 "https://$ROBOT_IP/desk/"

section "Docker installation"
run command -v docker
run docker --version

if command -v docker >/dev/null 2>&1; then
    section "Docker containers"
    run docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Networks}}'

    section "Likely FR3 Docker images"
    printf '$ docker image ls | grep FR3-related names\n'
    docker image ls --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}' 2>/dev/null \
        | grep -Ei 'fr3|franka|vineet' || true

    if docker inspect fr3py-vineet >/dev/null 2>&1; then
        section "fr3py-vineet container summary"
        run docker inspect --format \
            'name={{.Name}} image={{.Config.Image}} status={{.State.Status}} privileged={{.HostConfig.Privileged}} network_mode={{.HostConfig.NetworkMode}} created={{.Created}}' \
            fr3py-vineet
    fi
fi

section "Installed FR3 bridge executables"
for exe in fr3_joint_interface fr3_joint_velocity fr3_joint_torque fr3_task_interface; do
    path="$(command -v "$exe" 2>/dev/null || true)"
    printf '%s: %s\n' "$exe" "${path:-not found on host PATH}"
    if [ -n "$path" ]; then
        run file "$path"
        printf '$ ldd %q | grep franka/lcm\n' "$path"
        ldd "$path" 2>/dev/null | grep -Ei 'franka|lcm' || true
    fi
done

section "Common install locations"
find /usr/local/bin /usr/bin /opt "$HOME" -maxdepth 5 -type f \
    \( -name 'fr3_joint_interface' -o -name 'fr3_joint_velocity' \
       -o -name 'fr3_joint_torque' -o -name 'fr3_task_interface' \) \
    2>/dev/null | sort || true

section "System libfranka package"
run dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' libfranka
run readlink -f /usr/lib/libfranka.so

section "Realtime status"
run id
printf '$ ulimit -r\n'; ulimit -r || true
printf '$ ulimit -l\n'; ulimit -l || true
run grep -R -n -E '(@realtime|rtprio|memlock)' /etc/security/limits.conf /etc/security/limits.d

section "Relevant network details"
run ip -br address
run ip route

section "Complete"
printf 'Saved to: %s\n' "$OUTPUT"
printf 'No FCI/libfranka connection or robot command was issued.\n'
