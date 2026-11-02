#!/usr/bin/env bash
# Prints one line when a dataset's summaries are finished on the GPU VM, and a last line when the writer has stopped.
# Usage: VM_IP=<ip> VM_ID=<instance id> watch_summaries.sh      (polls every 60 s)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
seen=" "
while true; do
  if out=$("$here/eval/vm/vmssh.sh" 'ls ~/summaries_*.done 2>/dev/null; if pgrep -f run_summaries.sh > /dev/null; then echo RUNNING; else echo STOPPED; fi' 2>/dev/null); then
    while IFS= read -r line; do
      case "$line" in
        *summaries_*.done)
          name="${line##*summaries_}"
          name="${name%.done}"
          case "$seen" in
            *" $name "*) ;;
            *) echo "DONE $name"; seen="$seen$name " ;;
          esac
          ;;
      esac
    done <<< "$out"
    case "$out" in
      *STOPPED*) echo "STOPPED: the summary writer is not running any more"; exit 0 ;;
    esac
  else
    echo "ssh check failed"
  fi
  sleep 60
done
