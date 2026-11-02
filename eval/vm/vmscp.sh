#!/usr/bin/env bash
# Plain scp to/from the GPU VM (same environment as vmssh.sh).
#   VM_IP=... VM_ID=... eval/vm/vmscp.sh up LOCAL...            copies into the VM home directory
#   VM_IP=... VM_ID=... eval/vm/vmscp.sh down REMOTE_PATH LOCAL_DIR
set -euo pipefail
: "${VM_IP:?set VM_IP}" "${VM_ID:?set VM_ID}"
OPTS=(-i "$HOME/.ssh/google_compute_engine" -o CheckHostIP=no -o HashKnownHosts=no
  -o "HostKeyAlias=compute.${VM_ID}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
  -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" -o ConnectTimeout=20)
TARGET="${VM_USER:-$USER}@${VM_IP}"
mode="$1"; shift
if [ "$mode" = up ]; then
  exec /usr/bin/scp "${OPTS[@]}" "$@" "${TARGET}:~/"
else
  exec /usr/bin/scp "${OPTS[@]}" "${TARGET}:$1" "$2"
fi
