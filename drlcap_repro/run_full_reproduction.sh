#!/usr/bin/env bash
# Full DRLCap reproduction queue on the RTX 4070 Ti (GPU 1), one job at a
# time so measurements never overlap:
#   1. offline training (5000 iterations each): DRLCap frequency capping
#      (seeds 0 and 1), power capping, hybrid, and the tabular RL baseline
#   2. main comparison on all unseen test programs, 10 repeats (paper
#      Fig. 9/10): NVIDIA default, static caps, COORD-like, DRLCap and its
#      variants, RL baseline
#   3. ablations: second training seed, frozen network (no online
#      learning), monitoring-only overhead (Fig. 11)
# Every run writes its own timestamped results/ directory; a failure stops
# the queue (set -e) and gpu_control resets the GPU on exit.
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
LOG=results/queue_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

latest() { ls -d results/train_"$1"_2* | tail -1; }

echo "== training $(date)"
python drlcap.py train --variant freq   --agent ddqn    --seed 0 --tag freq_s0
python drlcap.py train --variant freq   --agent ddqn    --seed 1 --tag freq_s1
python drlcap.py train --variant power  --agent ddqn    --seed 0 --tag power_s0
python drlcap.py train --variant hybrid --agent ddqn    --seed 0 --tag hybrid_s0
python drlcap.py train --variant freq   --agent tabular --seed 0 --tag rl_s0

FREQ=$(latest freq_s0)/model_final.keras
FREQ1=$(latest freq_s1)/model_final.keras
POWER=$(latest power_s0)/model_final.keras
HYBRID=$(latest hybrid_s0)/model_final.keras
RL=$(latest rl_s0)/model_final.json

echo "== main comparison $(date)"
python evaluate.py --repeats 10 --static-caps 2100 2400 --coord \
    --models drlcap="$FREQ" drlcap_power="$POWER" \
             drlcap_hybrid="$HYBRID" rl="$RL" \
    --tag main

echo "== seed / overhead ablation $(date)"
python evaluate.py --repeats 5 \
    --models drlcap="$FREQ" drlcap_seed1="$FREQ1" --monitor-only drlcap \
    --tag ablation_seed_overhead

echo "== frozen-network ablation $(date)"
python evaluate.py --repeats 5 --no-online --models drlcap="$FREQ" \
    --tag ablation_frozen

echo "== done $(date)"
