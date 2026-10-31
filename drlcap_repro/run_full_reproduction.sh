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
#   4. tables and figures (report.py, analyze_eval.py, probe_policy.py,
#      plots.py)
# v3 (2026-10-31): reward attributed one sample later (drlcap.REWARD_DELAY)
# and NVML polled every 300 ms in every policy, baseline included.
# Every run writes its own timestamped results/ directory; a failure stops
# the queue (set -e) and gpu_control resets the GPU on exit.
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
V=v3
LOG=results/queue_${V}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

latest() { ls -d results/"$1"_2* | tail -1; }

echo "== training $(date)"
python drlcap.py train --variant freq   --agent ddqn    --seed 0 --tag freq_s0_$V
python drlcap.py train --variant freq   --agent ddqn    --seed 1 --tag freq_s1_$V
python drlcap.py train --variant power  --agent ddqn    --seed 0 --tag power_s0_$V
python drlcap.py train --variant hybrid --agent ddqn    --seed 0 --tag hybrid_s0_$V
python drlcap.py train --variant freq   --agent tabular --seed 0 --tag rl_s0_$V

T_FREQ=$(latest train_freq_s0_$V)
T_FREQ1=$(latest train_freq_s1_$V)
T_POWER=$(latest train_power_s0_$V)
T_HYBRID=$(latest train_hybrid_s0_$V)
T_RL=$(latest train_rl_s0_$V)
FREQ=$T_FREQ/model_final.keras

echo "== main comparison $(date)"
python evaluate.py --repeats 10 --static-caps 2100 2400 --coord \
    --models drlcap="$FREQ" drlcap_power="$T_POWER/model_final.keras" \
             drlcap_hybrid="$T_HYBRID/model_final.keras" \
             rl="$T_RL/model_final.json" \
    --tag main_$V

echo "== seed / overhead ablation $(date)"
python evaluate.py --repeats 5 \
    --models drlcap="$FREQ" drlcap_seed1="$T_FREQ1/model_final.keras" \
    --monitor-only drlcap --tag ablation_seed_overhead_$V

echo "== frozen-network ablation $(date)"
python evaluate.py --repeats 5 --no-online --models drlcap="$FREQ" \
    --tag ablation_frozen_$V

echo "== tables and figures $(date)"
MAIN=$(latest eval_main_$V)
ABL_SEED=$(latest eval_ablation_seed_overhead_$V)
ABL_FROZEN=$(latest eval_ablation_frozen_$V)
for d in "$MAIN" "$ABL_SEED" "$ABL_FROZEN"; do
    python report.py "$d"
    python analyze_eval.py "$d"
done
python probe_policy.py "$T_FREQ" "$T_FREQ1" "$T_POWER" "$T_HYBRID" \
    | tee results/probe_policy_${V}_$(date +%Y%m%d_%H%M%S).txt
python plots.py --main "$MAIN" --ablations "$ABL_SEED" "$ABL_FROZEN" \
    --train "$T_FREQ" "$T_FREQ1" "$T_POWER" "$T_HYBRID" "$T_RL" \
    --out results/plots_${V}_$(date +%Y%m%d_%H%M%S)

echo "== done $(date)"
