#!/usr/bin/env bash
# Resume of run_faithful.sh (V=f1) after the main comparison stopped on
# 2026-11-06 13:45 in repeat 9 (GROMACS refused a 100th md.log backup;
# fixed in workloads.py with GMX_MAXBACKUP=-1). Repeats 0-8 of
# results/eval_main_f1_20261104_210128 are complete; repeat 9 is run again
# here as its own one-repeat evaluation (same policies and settings, new
# COORD profile) and the 57 partial repeat-9 runs are not used. Stages 3
# and 4 are unchanged from run_faithful.sh.
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
V=f1
LOG=results/queue_${V}_resume_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

latest() { ls -d results/"$1"_2* | tail -1; }
FREQ=$(latest train_freq_s0_$V)/model_final.keras

echo "== 2b. main comparison, repeat 9 $(date)"
python evaluate.py --repeats 1 --static-caps 1800 2100 2400 2745 --coord \
    --models drlcap="$FREQ" \
             drlcap_power="$(latest train_power_s0_$V)/model_final.keras" \
             drlcap_hybrid="$(latest train_hybrid_s0_$V)/model_final.keras" \
             rl="$(latest train_rl_s0_$V)/model_final.json" \
    --tag main_rep9_$V
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
