#!/usr/bin/env bash
# Remove stale git lock files left behind by an interrupted git process.
# Safe to run anytime; does nothing if there are no locks.
cd "$(dirname "$0")" || exit 1
removed=0
for lock in .git/index.lock .git/HEAD.lock .git/config.lock .git/refs/heads/*.lock .git/refs/tags/*.lock; do
  if [ -e "$lock" ]; then
    rm -f "$lock" && echo "removed $lock" && removed=1
  fi
done
[ "$removed" -eq 0 ] && echo "no git locks found"
exit 0
