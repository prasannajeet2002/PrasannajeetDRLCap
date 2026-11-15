#!/usr/bin/env bash
# f2 = the f1 reference configuration with ONE setting changed: the
# minimum frequency cap is 2100 MHz instead of 1800 MHz (F2_CHANGES.md),
# plus a model-selection step on the validation programs. Everything else
# (Algorithm 1 / Eq. 1 reward, Table V hyper-parameters, 300 ms sampling,
# case-A forcing, reward delay 0, fresh model copy per test job with
# online learning and epsilon 0.1) is f1 unchanged. RTX 4070 Ti (GPU 1)
# only, one job at a time; f1 results are not touched.
#
#   1. offline training, 5000 iterations each: DRLCap frequency seeds
#      0-4, hybrid, tabular RL (same action candidates as DRLCap, as in
#      the paper). The power variant has no frequency cap, so its f1
#      model is the same configuration and is reused.
#   2. the five frequency seeds on the VALIDATION programs, 3 repeats;
#      the seed with the best geometric-mean EDP goes to the test
#   3. main comparison on the 10 Table II test programs, 5 repeats:
#      NVIDIA default, COORD-like, RL, DRLCap frequency (selected seed;
#      seed 0 as well if it was not the selected one), power, hybrid
#   4. the same comparison again, 5 more repeats (10 in total, as in the
#      paper)
#   5. selected model with a frozen network (no online learning), 3 repeats
# Training and evaluation wait between jobs whenever another user's job
# is on the GPU. A failure stops the queue (set -e); gpu_control resets
# the GPU on exit.
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
V=f2
CAPMIN=2100
SEEDS="0 1 2 3 4"
LOG=results/queue_${V}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

latest() { ls -d results/"$1"_2* | tail -1; }
wait_free() {  # the train / evaluate commands refuse to start on a busy GPU
    python - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import gpu_control as G
G.init()
while G.gpu_busy_by_others():
    time.sleep(30)
PY
}

echo "== 1. training $(date)"
for S in $SEEDS; do
    wait_free
    python drlcap.py train --cap-min $CAPMIN --variant freq --agent ddqn \
        --seed "$S" --tag freq_s${S}_$V
done
wait_free
python drlcap.py train --cap-min $CAPMIN --variant hybrid --agent ddqn \
    --seed 0 --tag hybrid_s0_$V
wait_free
python drlcap.py train --cap-min $CAPMIN --variant freq --agent tabular \
    --seed 0 --tag rl_s0_$V

echo "== 2. model selection on the validation programs $(date)"
VAL=$(python -c "import workloads as W; print(' '.join(W.VAL))")
MODELS=()
for S in $SEEDS; do
    MODELS+=("seed$S=$(latest train_freq_s${S}_$V)/model_final.keras")
done
wait_free
python evaluate.py --repeats 3 --workloads $VAL --models "${MODELS[@]}" \
    --tag val_$V
python select_model.py "$(latest eval_val_$V)"
SEL=$(python select_model.py "$(latest eval_val_$V)" | tail -1)
SEED0=$(latest train_freq_s0_$V)/model_final.keras
EXTRA=()
if [ "$SEL" != "$SEED0" ]; then EXTRA=("drlcap_seed0=$SEED0"); fi
MAIN=(drlcap="$SEL"
      drlcap_power="$(latest train_power_s0_f1)/model_final.keras"
      drlcap_hybrid="$(latest train_hybrid_s0_$V)/model_final.keras"
      rl="$(latest train_rl_s0_$V)/model_final.json"
      "${EXTRA[@]}")

echo "== 3. main comparison, repeats 1-5 $(date)"
wait_free
python evaluate.py --repeats 5 --coord --models "${MAIN[@]}" --tag main_a_$V

echo "== 4. main comparison, repeats 6-10 $(date)"
wait_free
python evaluate.py --repeats 5 --coord --models "${MAIN[@]}" --tag main_b_$V

echo "== 5. frozen network (no online learning) $(date)"
wait_free
python evaluate.py --repeats 3 --no-online --models drlcap="$SEL" \
    --tag frozen_$V

echo "== done $(date)"
