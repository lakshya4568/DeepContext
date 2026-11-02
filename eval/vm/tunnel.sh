#!/usr/bin/env bash
# Forwards the GPU VM's vLLM (port 8000) to local port 18000 and its reranker (port 8001) to local port 18001.
#   VM_IP=<external ip> VM_ID=<instance id> nohup eval/vm/tunnel.sh > tunnel.log 2>&1 &
# With TUNNEL_FOREVER=1 it reconnects by itself after a network break (5 s pause), until it is stopped.
# Stop it with `pkill -f 'ssh -N -L 18000'` (and the script itself). Health: curl -sf http://127.0.0.1:18000/health ; curl -sf http://127.0.0.1:18001/health
: "${VM_IP:?set VM_IP}" "${VM_ID:?set VM_ID}"
connect() {
  /usr/bin/ssh -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 \
    -i "$HOME/.ssh/google_compute_engine" \
    -o CheckHostIP=no -o HashKnownHosts=no -o "HostKeyAlias=compute.${VM_ID}" \
    -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    "${VM_USER:-$USER}@${VM_IP}"
}
if [ -n "${TUNNEL_FOREVER:-}" ]; then
  while true; do
    connect
    echo "tunnel closed ($(date +%H:%M:%S)), reconnecting in 5 s"
    sleep 5
  done
fi
exec /usr/bin/ssh -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 \
  -i "$HOME/.ssh/google_compute_engine" \
  -o CheckHostIP=no -o HashKnownHosts=no -o "HostKeyAlias=compute.${VM_ID}" \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
  "${VM_USER:-$USER}@${VM_IP}"
