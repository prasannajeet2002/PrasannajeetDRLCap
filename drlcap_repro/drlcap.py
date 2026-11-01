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

Variants (paper Table III): "freq" = frequency capping (DRLCap proper),
"power" = power capping, "hybrid" = frequency x power capping. Agents:
"ddqn" = DRLCap, "tabular" = the paper's "RL" baseline (double Q-learning
with two tables, same actions and reward; rl_baseline.py).

Offline training:
    python drlcap.py train [--variant freq] [--agent ddqn]
                           [--iterations 5000] [--tag NAME] [--seed 0]
"""

import os

# DRLCap runs on the host CPU; never let TensorFlow touch a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import resource
import threading
import time

import numpy as np

import agent as A
import gpu_control as G
import reward as R
import rl_baseline as T
import workloads as W

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SAMPLE_INTERVAL = 0.3

# Reward attribution delay in samples. NVML reports the effect of a new cap
# mostly one 300 ms sample late: in the v2 logs a cap change applied after
# sample t moves the clock by 0.29x in sample t -> t+1 but by 0.53x in
# t+1 -> t+2 (README). With delay 1 the reward of action a_t is computed on
# samples t+1 -> t+2 instead of t -> t+1 (which mostly showed a_(t-1)).
# The transition stored is still (s_t, a_t, r, s_(t+1)). 0 = old behaviour.
REWARD_DELAY = 1

# Algorithm 1 case A forces the maximum cap when a sample shows 100 % core
# and 0 % memory utilisation. On the 4070 Ti NVML reports 100 % whenever a
# kernel ran for the whole period, and the training-set sweep shows the
# maximum cap is the WORST EDP choice for the compute-bound programs
# (gaussian, hotspot, particlefilter: EDP-best 2200-2400 MHz). v4 disables
# the override (DEVIATION) and lets the v4 reward judge case A samples.
FORCE_CASE_A = False

# Action candidates: frequency caps in MHz (paper Table IV analogue),
# chosen from the frequency_sweep.py results on this 4070 Ti (README):
# under the default policy the SM clock peaks at ~2745 MHz; below 1800 MHz
# compute-bound jobs lose EDP (+15-60 %) and memory-bound jobs save no
# further energy. 1800/2745 = 0.66, close to the paper's V100S range
# 1035/1597 = 0.65. Step = 15 MHz hardware step -> 64 actions (V100S: ~72).
CAP_MIN = 1800
CAP_MAX = 2745
CAP_STEP = 15
FREQ_CAPS = list(range(CAP_MIN, CAP_MAX + 1, CAP_STEP))

# Power-capping variant: the whole range the driver accepts, 100-285 W in
# 5 W steps (38 actions). Measured Rodinia jobs draw only ~30-100 W at
# default clocks (README), so on most of them no power cap binds; the
# V100S scripts use 160-250 W of a 100-250 W range.
POWER_CAPS = list(range(G.MIN_POWER_LIMIT_W, int(G.POWER_LIMIT_W), 5)) \
    + [int(G.POWER_LIMIT_W)]

# Hybrid variant: 8 frequency caps x 8 power caps = 64 actions, the same
# number as the frequency variant (the paper notes the hybrid action space
# is large; it does not give its size).
HYBRID_FREQ = [1800, 1935, 2070, 2205, 2340, 2475, 2610, 2745]
HYBRID_POWER = [100, 126, 153, 179, 206, 232, 259, 285]

# Each action is (frequency cap MHz or None, power cap W or None); the
# last action is always "everything at maximum" (Algorithm 1 case A).
VARIANTS = {
    "freq": [(f, None) for f in FREQ_CAPS],
    "power": [(None, p) for p in POWER_CAPS],
    "hybrid": [(f, p) for p in HYBRID_POWER for f in HYBRID_FREQ],
    # v4: the frequency variant on the hybrid's 8-cap ladder (135 MHz
    # apart) so each action gets ~8x more training samples
    "freq8": [(f, None) for f in HYBRID_FREQ],
    # v4 power and hybrid variants with as few actions per knob (8 power
    # caps; 4 x 4 = 16 hybrid pairs) for the same reason
    "power8": [(None, p) for p in HYBRID_POWER],
    "hybrid16": [(f, p) for p in HYBRID_POWER[::2] + [HYBRID_POWER[-1]]
                 if p != HYBRID_POWER[-2]
                 for f in HYBRID_FREQ[::2] + [HYBRID_FREQ[-1]]
                 if f != HYBRID_FREQ[-2]],
}
ACTIONS = VARIANTS["freq"]


def primary(action):
    """The cap value used by Algorithm 1 case B (c / action)."""
    f, p = action
    return f if f is not None else p


# State discretisation (paper Sec. IV-D "preprocessed and discretized";
# bucket layout from the authors' V100S code, ranges adapted to the
# 4070 Ti): (lo, hi, buckets) per metric; the SM clock becomes its index
# on the 15 MHz hardware ladder 210..3105 MHz. Bucket indices are divided
# by the bucket count so the network sees values in [0, 1) (the authors
# feed raw indices; scaling is our choice for numerical stability).
BUCKETS = {
    "util": (0.0, 100.0, 20),
    "mem_util": (0.0, 100.0, 20),
    "power": (10.0, G.POWER_LIMIT_W, 20),
    "temp": (30.0, 90.0, 30),
}
CLOCK_LEVELS = (G.MAX_CLOCK_MHZ - G.MIN_CLOCK_MHZ) // 15 + 1

STATE_SIZE = 5

LOG_FIELDS = ["job", "workload", "t", "util", "mem_util", "power", "clock",
              "temp", "case", "reward", "action", "cap_mhz", "power_cap_w",
              "explore",
              "forced_max", "epsilon", "updates", "loss"]


def to_state(sample):
    """Discretised 5-feature state (paper Sec. IV-B, IV-D), in [0, 1)."""
    feats = []
    for key in ("util", "mem_util", "power", "temp"):
        lo, hi, n = BUCKETS[key]
        b = int(np.clip((sample[key] - lo) / ((hi - lo) / n), 0, n - 1))
        feats.append(b / n)
    level = int(np.clip(round((sample["clock"] - G.MIN_CLOCK_MHZ) / 15),
                        0, CLOCK_LEVELS - 1))
    feats.append(level / CLOCK_LEVELS)
    # order as in the paper: util, mem util, power, frequency, temperature
    u, m, p, t, f = feats
    return np.array([u, m, p, f, t], dtype=np.float32)


def make_agent(kind, num_actions, seed):
    if kind == "ddqn":
        return A.DDQNAgent(STATE_SIZE, num_actions, seed=seed)
    if kind == "tabular":
        return T.DoubleQTable(num_actions, seed=seed)
    raise ValueError(kind)


def run_job(agent, workload, *, learn, actions=None,
            epsilon_fn=None, greedy=False, log=None, job_id=0,
            apply_caps=True):
    """Run one workload job under DRLCap control.

    learn       update the network online (offline training and the
                paper's online deployment both do this)
    epsilon_fn  called each step to set epsilon (offline schedule);
                None keeps agent.epsilon fixed (online deployment)
    apply_caps  False runs the whole loop (sampling, reward, learning,
                action choice) but never touches the GPU knobs: the
                monitoring-overhead experiment (paper Fig. 11)
    Returns a dict with runtime, energy, EDP and control statistics.
    """
    actions = actions or ACTIONS
    c_idle = float(min(primary(a) for a in actions))
    power_sum, power_n = 0.0, 0  # running mean board power (v4 reward)
    result = {}

    def target():
        try:
            result["time_s"] = W.run_blocking(workload)
        except Exception as exc:  # surfaced after the loop
            result["error"] = exc

    G.reset_cap()
    applied = (None, None)
    prev = None
    # (state, action) of the last REWARD_DELAY + 1 decisions, oldest first,
    # and the states that followed them; the oldest gets the reward of the
    # newest sample pair
    pending, caps_used, steps, compute_ms = [], [], 0, []

    cpu0 = time.process_time()
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
            power_sum += cur["power"]
            power_n += 1

            if pending:
                pending[-1][2] = state  # next state of the newest decision
            if prev is not None and len(pending) > REWARD_DELAY:
                s_old, a_old, s_next = pending.pop(0)
                r, case = R.compute(prev, cur, primary(actions[a_old]),
                                    c_idle, power_sum / power_n)
                agent.remember(s_old, a_old, r, s_next, done)
                if learn:
                    agent.learn()
            if done:
                break

            if epsilon_fn is not None:
                epsilon_fn()
            action, explored = agent.act(state, greedy=greedy)
            forced = FORCE_CASE_A and case == "A"
            if forced:  # Algorithm 1 line 7: compute-bound -> max cap
                action = len(actions) - 1
            cap = actions[action]
            if apply_caps and cap != applied:
                G.apply(cap, applied)
                applied = cap

            if log is not None:
                log.writerow({
                    "job": job_id, "workload": workload,
                    "t": round(time.perf_counter(), 3), **{
                        k: cur[k] for k in
                        ("util", "mem_util", "power", "clock", "temp")},
                    "case": case,
                    "reward": "" if r is None else round(r, 4),
                    "action": action, "cap_mhz": cap[0] or "",
                    "power_cap_w": cap[1] or "",
                    "explore": int(explored), "forced_max": int(forced),
                    "epsilon": round(agent.epsilon, 4),
                    "updates": agent.updates,
                    "loss": round(agent.last_loss, 6),
                })

            prev = cur
            pending.append([state, action, None])
            caps_used.append(cap)
            steps += 1
            compute_ms.append(1000 * (time.perf_counter() - tick))
            time.sleep(max(0.0, SAMPLE_INTERVAL - (time.perf_counter() - tick)))
    finally:
        G.reset_cap()
    runner.join()
    if "error" in result:
        raise result["error"]

    energy = G.energy_j() - e0
    cpu_s = time.process_time() - cpu0
    return {
        "workload": workload,
        "time_s": result["time_s"],
        "energy_j": energy,
        "edp": energy * result["time_s"],
        "steps": steps,
        "mean_cap_mhz": _mean([c[0] for c in caps_used]),
        "mean_power_cap_w": _mean([c[1] for c in caps_used]),
        # controller overhead (paper Sec. VI-C): CPU seconds used by this
        # process during the job, per-step compute time (sample + reward +
        # learn + act + apply), and peak resident memory
        "agent_cpu_s": round(cpu_s, 3),
        "step_ms_median": round(float(np.median(compute_ms)), 2)
        if compute_ms else "",
        "step_ms_p95": round(float(np.percentile(compute_ms, 95)), 2)
        if compute_ms else "",
        "max_rss_mb": round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else ""


def train(iterations, tag, seed, variant="freq", kind="ddqn"):
    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS, f"train_{tag}_{stamp}")
    os.makedirs(run_dir)

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")

    actions = VARIANTS[variant]
    agent = make_agent(kind, len(actions), seed)
    rng = np.random.default_rng(seed)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "variant": variant, "agent": kind, "actions": actions,
            "buckets": BUCKETS, "sample_interval_s": SAMPLE_INTERVAL,
            "reward_delay_samples": REWARD_DELAY,
            "reward_mode": R.REWARD_MODE, "reward_scale": R.REWARD_SCALE,
            "force_case_a": FORCE_CASE_A,
            "iterations": iterations, "seed": seed, "train_set": W.TRAIN,
            "hyper": {k: getattr(A, k) for k in (
                "BATCH_SIZE", "LEARNING_RATE", "BUFFER_SIZE", "GAMMA",
                "EPSILON_START", "EPSILON_END", "TARGET_UPDATE",
                "HIDDEN_UNITS", "PER_ALPHA", "PER_BETA_START",
                "PER_BETA_INCREMENT", "PER_EPS", "PER_ABS_ERR_UPPER",
                "LEARN_START_OFFLINE", "LEARN_START_ONLINE")},
            "tabular": {k: getattr(T, k) for k in ("ALPHA", "GAMMA")},
            "reward": {k: getattr(R, k) for k in (
                "STEP_MHZ", "N", "G_POWER_DOWN", "G_POWER_UP",
                "P_THRESH_W", "R_CASE_A", "R_IDLE_MAX", "REWARD_CLIP")},
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

    job_id = 0
    try:
        while progress["step"] < iterations:
            for workload in rng.permutation(W.TRAIN):
                if progress["step"] >= iterations:
                    break
                t0 = time.time()
                res = run_job(agent, str(workload), learn=True,
                              actions=actions, epsilon_fn=eps_schedule,
                              log=steps, job_id=job_id)
                res.update(job=job_id, step=progress["step"],
                           epsilon=agent.epsilon, updates=agent.updates)
                jobs.append(res)
                with open(jobs_path, "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(jobs[0]))
                    wr.writeheader()
                    wr.writerows(jobs)
                step_log.flush()
                agent.save(os.path.join(run_dir, "model_latest" + agent.EXT))
                print(f"job {job_id:3d} {workload:15s} "
                      f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J "
                      f"cap={res['mean_cap_mhz']}/{res['mean_power_cap_w']} "
                      f"step={progress['step']}/{iterations} "
                      f"eps={agent.epsilon:.3f} updates={agent.updates} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                job_id += 1
    finally:
        G.reset_cap()
        step_log.close()
    agent.save(os.path.join(run_dir, "model_final" + agent.EXT))
    print("saved", run_dir)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--iterations", type=int, default=A.TRAINING_ITERATIONS)
    t.add_argument("--tag", default="drlcap")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--variant", choices=sorted(VARIANTS), default="freq")
    t.add_argument("--agent", choices=["ddqn", "tabular"], default="ddqn")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.iterations, args.tag, args.seed, args.variant, args.agent)


if __name__ == "__main__":
    main()
