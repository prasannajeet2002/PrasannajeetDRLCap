"""DRLCap runtime controller and offline training (paper Sec. IV, Alg. 2).

Control loop, every SAMPLE_INTERVAL (300 ms, paper Sec. VI-C):
  1. workload monitor reads util, mem util, power, clock, temperature
  2. reward for the previous action from Algorithm 1 (reward.py)
  3. store (s, a, r, s', is_end) in prioritized replay, one DDQN update
  4. epsilon-greedy action = frequency cap; in Algorithm 1 case A the cap
     is forced to the maximum
  5. power controller applies the cap only if it changed

The agent is a CPU process (CUDA hidden from TensorFlow); the workload is
an unmodified program on the RTX 4070 Ti.

Offline training:
    python drlcap.py train [--iterations 5000] [--tag NAME]
"""

import os

# DRLCap runs on the host CPU; never let TensorFlow touch a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import threading
import time

import numpy as np

import agent as A
import gpu_control as G
import reward as R
import workloads as W

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SAMPLE_INTERVAL = 0.3

# Action candidates: frequency caps in MHz (paper Table IV analogue),
# chosen from the frequency_sweep.py results on this 4070 Ti (README):
# under the default policy the SM clock peaks at ~2745 MHz; below 1800 MHz
# compute-bound jobs lose EDP (+15-60 %) and memory-bound jobs save no
# further energy. 1800/2745 = 0.66, close to the paper's V100S range
# 1035/1597 = 0.65. Step = 15 MHz hardware step -> 64 actions (V100S: ~72).
CAP_MIN = 1800
CAP_MAX = 2745
CAP_STEP = 15
ACTIONS = list(range(CAP_MIN, CAP_MAX + 1, CAP_STEP))

STATE_SIZE = 5

LOG_FIELDS = ["job", "workload", "t", "util", "mem_util", "power", "clock",
              "temp", "case", "reward", "action", "cap_mhz", "explore",
              "forced_max", "epsilon", "updates", "loss"]


def to_state(sample):
    """Normalised 5-feature state (paper Sec. IV-B)."""
    return np.array([
        sample["util"] / 100.0,
        sample["mem_util"] / 100.0,
        sample["power"] / G.POWER_LIMIT_W,
        sample["clock"] / G.MAX_CLOCK_MHZ,
        sample["temp"] / 100.0,
    ], dtype=np.float32)


def run_job(agent, workload, *, learn, beta_fn=lambda: 1.0,
            epsilon_fn=None, greedy=False, log=None, job_id=0):
    """Run one workload job under DRLCap control.

    learn       update the network online (offline training and the
                paper's online deployment both do this)
    epsilon_fn  called each step to set epsilon (offline schedule);
                None keeps agent.epsilon fixed (online deployment)
    Returns a dict with runtime, energy, EDP and control statistics.
    """
    c_idle = float(ACTIONS[0])
    result = {}

    def target():
        try:
            result["time_s"] = W.run_blocking(workload)
        except Exception as exc:  # surfaced after the loop
            result["error"] = exc

    G.reset_cap()
    applied = None
    prev, prev_state, prev_action = None, None, None
    caps_used, steps = [], 0

    e0 = G.energy_j()
    runner = threading.Thread(target=target, daemon=True)
    runner.start()
    try:
        while True:
            tick = time.perf_counter()
            done = not runner.is_alive()
            cur = G.read()
            state = to_state(cur)
            r, case = (None, R.case_of(cur))

            if prev is not None:
                r, case = R.reward(prev, cur, ACTIONS[prev_action], c_idle)
                agent.remember(prev_state, prev_action, r, state, done)
                if learn:
                    agent.learn(beta_fn())
            if done:
                break

            if epsilon_fn is not None:
                epsilon_fn()
            action, explored = agent.act(state, greedy=greedy)
            forced = case == "A"
            if forced:  # Algorithm 1 line 7: compute-bound -> max cap
                action = len(ACTIONS) - 1
            cap = ACTIONS[action]
            if cap != applied:
                G.set_cap(cap)
                applied = cap

            if log is not None:
                log.writerow({
                    "job": job_id, "workload": workload,
                    "t": round(time.perf_counter(), 3), **{
                        k: cur[k] for k in
                        ("util", "mem_util", "power", "clock", "temp")},
                    "case": case,
                    "reward": "" if r is None else round(r, 4),
                    "action": action, "cap_mhz": cap,
                    "explore": int(explored), "forced_max": int(forced),
                    "epsilon": round(agent.epsilon, 4),
                    "updates": agent.updates,
                    "loss": round(agent.last_loss, 6),
                })

            prev, prev_state, prev_action = cur, state, action
            caps_used.append(cap)
            steps += 1
            time.sleep(max(0.0, SAMPLE_INTERVAL - (time.perf_counter() - tick)))
    finally:
        G.reset_cap()
    runner.join()
    if "error" in result:
        raise result["error"]

    energy = G.energy_j() - e0
    return {
        "workload": workload,
        "time_s": result["time_s"],
        "energy_j": energy,
        "edp": energy * result["time_s"],
        "steps": steps,
        "mean_cap_mhz": float(np.mean(caps_used)) if caps_used else None,
    }


def train(iterations, tag, seed):
    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS, f"train_{tag}_{stamp}")
    os.makedirs(run_dir)

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")

    agent = A.DDQNAgent(STATE_SIZE, len(ACTIONS), seed=seed)
    rng = np.random.default_rng(seed)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "actions_mhz": ACTIONS, "sample_interval_s": SAMPLE_INTERVAL,
            "iterations": iterations, "seed": seed, "train_set": W.TRAIN,
            "hyper": {k: getattr(A, k) for k in (
                "BATCH_SIZE", "LEARNING_RATE", "BUFFER_SIZE", "GAMMA",
                "EPSILON_START", "EPSILON_END", "TARGET_UPDATE",
                "HIDDEN_UNITS", "PER_ALPHA", "PER_BETA_START")},
            "reward": {k: getattr(R, k) for k in (
                "STEP_MHZ", "N", "G_POWER_DOWN", "G_POWER_UP",
                "P_THRESH_W", "R_CASE_A", "REWARD_CLIP")},
        }, f, indent=2)

    step_log = open(os.path.join(run_dir, "steps.csv"), "w", newline="")
    steps = csv.DictWriter(step_log, fieldnames=LOG_FIELDS)
    steps.writeheader()
    jobs_path = os.path.join(run_dir, "jobs.csv")
    jobs = []

    progress = {"step": 0}

    def eps_schedule():
        agent.set_epsilon_by_progress(progress["step"], iterations)
        progress["step"] += 1

    def beta():
        frac = min(1.0, progress["step"] / iterations)
        return A.PER_BETA_START + frac * (1.0 - A.PER_BETA_START)

    job_id = 0
    try:
        while progress["step"] < iterations:
            for workload in rng.permutation(W.TRAIN):
                if progress["step"] >= iterations:
                    break
                t0 = time.time()
                res = run_job(agent, str(workload), learn=True, beta_fn=beta,
                              epsilon_fn=eps_schedule, log=steps,
                              job_id=job_id)
                res.update(job=job_id, step=progress["step"],
                           epsilon=agent.epsilon, updates=agent.updates)
                jobs.append(res)
                with open(jobs_path, "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(jobs[0]))
                    wr.writeheader()
                    wr.writerows(jobs)
                step_log.flush()
                agent.save(os.path.join(run_dir, "model_latest.keras"))
                print(f"job {job_id:3d} {workload:15s} "
                      f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J "
                      f"cap={res['mean_cap_mhz']:.0f} "
                      f"step={progress['step']}/{iterations} "
                      f"eps={agent.epsilon:.3f} updates={agent.updates} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                job_id += 1
    finally:
        G.reset_cap()
        step_log.close()
    agent.save(os.path.join(run_dir, "model_final.keras"))
    print("saved", run_dir)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--iterations", type=int, default=A.TRAINING_ITERATIONS)
    t.add_argument("--tag", default="drlcap")
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.iterations, args.tag, args.seed)


if __name__ == "__main__":
    main()
