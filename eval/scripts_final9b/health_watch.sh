#!/usr/bin/env bash
# One line per event while the final round runs: a dataset's summaries finished, the VM or the tunnel stops answering,
# the Mac disk runs low, the VM's 8-hour limit comes near, and a status line every 30 minutes. Ends when the summary
# writer on the VM has stopped. The limit time is the VM start plus 8 hours (VM started 2026-11-02 12:45 IST).
# With EXIT_ON_EVENT=1 it stays silent (no status lines, sets finished before the start are not reported) and ends
# at the first event, so that one wake-up means "something happened".
# Usage: VM_IP=<ip> VM_ID=<instance id> [WATCH_PORTS="18000"] [EXIT_ON_EVENT=1] health_watch.sh
#   (phase 1: only the 9B port, the reranker is not loaded)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
seen=" "
tunnel_down=" "
low_disk=0
warned_60=0
warned_20=0
tick=0
first=1
status_line=""
ssh_fail=0

emit() {
  echo "$*"
  if [ -n "${EXIT_ON_EVENT:-}" ]; then exit 0; fi
}

while true; do
  if out=$("$here/eval/vm/vmssh.sh" 'ls ~/summaries_*.done 2>/dev/null; if pgrep -f run_summaries.sh > /dev/null; then echo RUNNING; else echo STOPPED; fi; tail -n 1 ~/slm/summaries.out | cut -c1-90' 2> /dev/null); then
    while IFS= read -r line; do
      case "$line" in
        *summaries_*.done)
          name="${line##*summaries_}"
          name="${name%.done}"
          case "$seen" in
            *" $name "*) ;;
            *)
              seen="$seen$name "
              if [ "$first" -eq 0 ] || [ -z "${EXIT_ON_EVENT:-}" ]; then emit "DONE summaries of $name"; fi
              ;;
          esac
          ;;
      esac
    done <<< "$out"
    status_line="$(printf '%s\n' "$out" | tail -n 1)"
    ssh_fail=0
  else
    ssh_fail=$((ssh_fail + 1))
    if [ "$ssh_fail" -ge 2 ]; then emit "VM does not answer on ssh (two checks in a row)"; fi
  fi
  for port in ${WATCH_PORTS:-18000 18001}; do
    if curl -sf -m 5 "http://127.0.0.1:${port}/health" > /dev/null; then
      tunnel_down="${tunnel_down// $port /}"
    else
      case "$tunnel_down" in
        *" $port "*) ;;
        *) tunnel_down="$tunnel_down$port "; emit "NO ANSWER on tunnel port $port (18000 = 9B server, 18001 = reranker)" ;;
      esac
    fi
  done
  free_gb=$(df -k /System/Volumes/Data | awk 'NR==2 {printf "%.1f", $4 / 1048576}')
  if [ "${free_gb%.*}" -lt 2 ] && [ "$low_disk" -eq 0 ]; then low_disk=1; emit "LOW DISK: ${free_gb} GB free on the Mac"; fi
  now=$(date +%H%M)
  if [ "$now" -ge 1945 ] && [ "$now" -lt 2300 ] && [ "$warned_60" -eq 0 ]; then warned_60=1; emit "VM stops itself at about 20:45 (under 60 minutes left)"; fi
  if [ "$now" -ge 2025 ] && [ "$now" -lt 2300 ] && [ "$warned_20" -eq 0 ]; then warned_20=1; emit "VM stops itself at about 20:45 (about 20 minutes left)"; fi
  tick=$((tick + 1))
  if [ -z "${EXIT_ON_EVENT:-}" ] && [ $((tick % 30)) -eq 0 ]; then echo "status $(date +%H:%M): ${status_line} | Mac disk free ${free_gb} GB"; fi
  first=0
  case "$out" in
    *STOPPED*) emit "STOPPED: the summary writer is not running any more"; exit 0 ;;
  esac
  sleep 60
done
