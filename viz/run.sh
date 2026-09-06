#!/usr/bin/env bash
# Start (or attach to) a tmux session running the Franka cuRobo UI.
set -euo pipefail

SESSION="franky_ui"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="source \$HOME/miniforge3/etc/profile.d/conda.sh && conda activate franky && export CUDA_HOME=/usr/local/cuda && cd '$DIR' && exec python server.py"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
else
    tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""
    echo "Started tmux session '$SESSION'."
fi

if [[ -t 0 && -t 1 ]]; then
    tmux attach -t "$SESSION"
fi
