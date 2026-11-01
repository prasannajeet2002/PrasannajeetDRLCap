"""Replay logged transitions of a training run through a fresh DDQN with
different gamma; report greedy actions on typical states."""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/home/drl/project/drlcap/drlcap_repro")
import numpy as np
import agent as A, drlcap as D, probe_policy as P
T, variant, gamma, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
A.GAMMA = gamma
acts = D.VARIANTS[variant]
rows = list(csv.DictReader(open(T + "/steps.csv")))
st = lambda r: D.to_state({k: float(r[k]) for k in ("util", "mem_util", "power", "clock", "temp")})
ag = A.DDQNAgent(D.STATE_SIZE, len(acts), seed=seed)
for k in range(2, len(rows)):
    a, b, c = rows[k-2], rows[k-1], rows[k]
    if a["job"] != c["job"] or not c["reward"]:
        continue
    done = k + 1 >= len(rows) or rows[k+1]["job"] != c["job"]
    ag.remember(st(a), int(a["action"]), float(c["reward"]), st(b), done)
    ag.learn()
typ = np.array([D.to_state(dict(x, temp=50.0, energy_j=0.0)) for _, x in P.TYPICAL], dtype=np.float32)
q = ag.q(typ).numpy()
print(f"gamma={gamma} seed={seed}: " + ", ".join(f"{n}->{D.primary(acts[i])}" for (n, _), i in zip(P.TYPICAL, q.argmax(1))))
