#!/usr/bin/env bash
# Paper-faithful DRLCap reproduction queue on the RTX 4070 Ti (GPU 1), one
# job at a time so measurements never overlap. Reference configuration =
# REFERENCE_CONFIG.md (paper Alg. 1 / Eq. 1 reward, gamma 0.95, Case-A
# forcing, reward delay 0, 64 frequency caps); training on the 64 training
# programs of Rodinia, Parboil, SHOC, PolyBench and the CUDA SDK; testing on
# the 10 Table II programs (workloads.py).
#
#   1. offline training, 5000 iterations each (paper Table V):
#      DRLCap frequency (seeds 0, 1, 2), power, hybrid, tabular RL
#   2. main comparison, 10 repeats (paper Fig. 9/10; review item 7):
#      NVIDIA default, static caps from the action set (review item 4),
#      COORD-like, DRLCap frequency / power / hybrid, RL
#   3. seeds, frozen network (no online learning), monitoring overhead,
#      5 repeats
#   4. the v4 settings, one at a time on top of the reference (review
#      item 2), trained and evaluated with 5 repeats
# Every run writes its own timestamped results/ directory; a failure stops
# the queue (set -e) and gpu_control resets the GPU on exit.
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
V=f1
LOG=results/queue_${V}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

latest() { ls -d results/"$1"_2* | tail -1; }

echo "== 1. training $(date)"
python drlcap.py train --variant freq   --agent ddqn    --seed 0 --tag freq_s0_$V
python drlcap.py train --variant power  --agent ddqn    --seed 0 --tag power_s0_$V
python drlcap.py train --variant hybrid --agent ddqn    --seed 0 --tag hybrid_s0_$V
python drlcap.py train --variant freq   --agent tabular --seed 0 --tag rl_s0_$V
python drlcap.py train --variant freq   --agent ddqn    --seed 1 --tag freq_s1_$V
python drlcap.py train --variant freq   --agent ddqn    --seed 2 --tag freq_s2_$V

FREQ=$(latest train_freq_s0_$V)/model_final.keras

echo "== 2. main comparison $(date)"
python evaluate.py --repeats 10 --static-caps 1800 2100 2400 2745 --coord \
    --models drlcap="$FREQ" \
             drlcap_power="$(latest train_power_s0_$V)/model_final.keras" \
             drlcap_hybrid="$(latest train_hybrid_s0_$V)/model_final.keras" \
             rl="$(latest train_rl_s0_$V)/model_final.json" \
    --tag main_$V

echo "== 3a. seeds and monitoring overhead $(date)"
python evaluate.py --repeats 5 \
    --models drlcap="$FREQ" \
             drlcap_seed1="$(latest train_freq_s1_$V)/model_final.keras" \
             drlcap_seed2="$(latest train_freq_s2_$V)/model_final.keras" \
    --monitor-only drlcap --tag seeds_overhead_$V

echo "== 3b. frozen network (no online learning) $(date)"
python evaluate.py --repeats 5 --no-online --models drlcap="$FREQ" \
    --tag frozen_$V

echo "== 4. v4 settings one at a time $(date)"
python drlcap.py train --seed 0 --reward-mode edp_local --tag var_edplocal_$V
python drlcap.py train --seed 0 --gamma 0.2             --tag var_gamma02_$V
python drlcap.py train --seed 0 --no-force-case-a       --tag var_nocasea_$V
python drlcap.py train --seed 0 --reward-delay 1        --tag var_delay1_$V
python drlcap.py train --seed 0 --variant freq8         --tag var_freq8_$V
python evaluate.py --repeats 5 --static-caps 1800 2745 \
    --models drlcap="$FREQ" \
             var_edplocal="$(latest train_var_edplocal_$V)/model_final.keras" \
             var_gamma02="$(latest train_var_gamma02_$V)/model_final.keras" \
             var_nocasea="$(latest train_var_nocasea_$V)/model_final.keras" \
             var_delay1="$(latest train_var_delay1_$V)/model_final.keras" \
             var_freq8="$(latest train_var_freq8_$V)/model_final.keras" \
    --tag variants_$V

echo "== done $(date)"
