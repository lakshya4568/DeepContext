#!/usr/bin/env bash
# Plain ssh to the GPU VM using the key gcloud created.
#   VM_IP=<external ip> VM_ID=<instance id> eval/vm/vmssh.sh 'remote command'
# Both values come from `gcloud compute instances describe` (networkInterfaces[0].accessConfigs[0].natIP and id).
# The IP changes on every start. (`gcloud compute ssh` adds a fresh key to the project metadata on every call, so
# use this wrapper for repeated calls.)
: "${VM_IP:?set VM_IP}" "${VM_ID:?set VM_ID}"
exec /usr/bin/ssh -T -i "$HOME/.ssh/google_compute_engine" \
  -o CheckHostIP=no -o HashKnownHosts=no -o "HostKeyAlias=compute.${VM_ID}" \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts" \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
  "${VM_USER:-$USER}@${VM_IP}" -- "$@"
