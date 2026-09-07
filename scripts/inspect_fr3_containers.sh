#!/usr/bin/env bash
# Read-only inspection of existing FR3 Docker containers and bridge binaries.
# Does not open a robot connection or issue any robot command.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-$REPO_ROOT/rt_fr3_containers_${TIMESTAMP}.txt}"
HOST_BRIDGE="${HOST_BRIDGE:-$HOME/FR3Py/fr3_bridge/build/src/fr3_joint_interface}"

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
printf 'Output: %s\n' "$OUTPUT"

section "Host bridge binary"
if [ -f "$HOST_BRIDGE" ]; then
    run stat -c 'path=%n size=%s modified=%y owner=%U:%G mode=%A' "$HOST_BRIDGE"
    run file "$HOST_BRIDGE"
    run ldd "$HOST_BRIDGE"
else
    printf 'Not found: %s\n' "$HOST_BRIDGE"
fi

section "Running FR3-related containers"
mapfile -t CONTAINERS < <(
    docker ps --format '{{.Names}}' 2>/dev/null | grep -Ei 'fr3|franka' || true
)
if [ "${#CONTAINERS[@]}" -eq 0 ]; then
    printf 'No running container names matched fr3/franka.\n'
fi

for container in "${CONTAINERS[@]}"; do
    section "Container: $container"
    run docker inspect --format \
        'name={{.Name}} image={{.Config.Image}} image_id={{.Image}} status={{.State.Status}} pid={{.State.Pid}} privileged={{.HostConfig.Privileged}} network_mode={{.HostConfig.NetworkMode}} created={{.Created}} started={{.State.StartedAt}}' \
        "$container"

    printf '\n--- Processes ---\n'
    run docker top "$container" -eo pid,ppid,user,stat,lstart,cmd

    printf '\n--- OS and Python ---\n'
    run docker exec "$container" sh -lc \
        'printf "os: "; grep -E "^(PRETTY_NAME|VERSION_ID)=" /etc/os-release 2>/dev/null | tr "\n" " "; echo; uname -r; command -v python || true; command -v python3 || true; python --version 2>&1 || true; python3 --version 2>&1 || true'

    printf '\n--- libfranka package and linker entries ---\n'
    run docker exec "$container" sh -lc \
        'dpkg-query -W -f="${Package}\t${Version}\t${Architecture}\n" libfranka 2>/dev/null || true; ldconfig -p 2>/dev/null | grep -Ei "libfranka|franky" || true'

    printf '\n--- Python franky packages ---\n'
    run docker exec "$container" sh -lc \
        'for py in python python3; do if command -v "$py" >/dev/null 2>&1; then echo "[$py]"; "$py" -m pip show franky-control 2>/dev/null || true; "$py" -m pip show fr3py 2>/dev/null || true; fi; done'

    printf '\n--- Bridge executables and linkage ---\n'
    run docker exec "$container" sh -lc \
        'for name in fr3_joint_interface fr3_joint_velocity fr3_joint_torque fr3_task_interface; do p=$(command -v "$name" 2>/dev/null || true); echo "$name: ${p:-not found}"; if [ -n "$p" ]; then file "$p" 2>/dev/null || true; ldd "$p" 2>/dev/null | grep -Ei "franka|lcm" || true; fi; done'
done

section "Docker image metadata for matching images"
while IFS=$'\t' read -r repository tag image_id; do
    [ -n "${image_id:-}" ] || continue
    printf '\n--- %s:%s (%s) ---\n' "$repository" "$tag" "$image_id"
    run docker image inspect --format \
        'id={{.Id}} created={{.Created}} architecture={{.Architecture}} os={{.Os}}' \
        "$image_id"
done < <(docker image ls --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' 2>/dev/null \
    | grep -Ei 'fr3|franka' || true)

section "Complete"
printf 'Saved to: %s\n' "$OUTPUT"
printf 'No FCI/libfranka connection or robot command was issued.\n'
