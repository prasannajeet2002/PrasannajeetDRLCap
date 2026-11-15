#!/usr/bin/env bash
# Smoke test of the f2 code on GPU 1, then the full queue (run_f2.sh) only
# if the smoke test and its checks pass. Run detached:
#   setsid nohup bash launch_f2.sh > results/launch_f2.log 2>&1 &
set -euo pipefail
source /home/drl/miniconda3/etc/profile.d/conda.sh
conda activate prasannajeet_env
cd "$(dirname "$0")"
latest() { ls -d results/"$1"_2* | tail -1; }
python - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import gpu_control as G
G.init()
while G.gpu_busy_by_others():
    time.sleep(30)
PY
echo "== smoke train $(date)"
python drlcap.py train --cap-min 2100 --iterations 150 --seed 0 --tag smoke_f2
echo "== smoke eval $(date)"
python evaluate.py --repeats 1 --workloads lud pb-histo \
    --models new="$(latest train_smoke_f2)/model_final.keras" \
             old="$(latest train_freq_s0_f1)/model_final.keras" \
    --tag smoke_f2
python select_model.py "$(latest eval_smoke_f2)"
python - "$(latest eval_smoke_f2)" <<'PY'
import csv, sys, os
d = sys.argv[1]
runs = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
pol = {r["job"]: r["policy"] for r in runs}
caps = {"new": [], "old": []}
for r in csv.DictReader(open(os.path.join(d, "steps.csv"))):
    if r["cap_mhz"]:
        caps[pol[r["job"]]].append(int(r["cap_mhz"]))
for k, lo in (("new", 2100), ("old", 1800)):
    c = caps[k]
    print(k, "steps", len(c), "caps", min(c), "-", max(c))
    assert len(c) > 20 and lo <= min(c) and max(c) <= 2745, k
assert min(caps["old"]) < 2100, "f1 model should still use its 1800-2745 range"
assert len(runs) == 6 and all(float(r["energy_j"]) > 0 for r in runs)
print("SMOKE OK")
PY
echo "== smoke passed, starting the f2 queue $(date)"
exec bash run_f2.sh
