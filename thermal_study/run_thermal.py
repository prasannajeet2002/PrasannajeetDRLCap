"""Thermal experiments on the RTX 4070 Ti (GPU 1), logged every 300 ms.

  leak   Leakage and thermal time constant. The GPU idles until it is cool,
         then a steady compute load (burn) runs for --burn-s seconds at a
         fixed clock cap, then the GPU idles again (cool-down). At a fixed
         clock and constant load, any rise in power while the die heats is
         leakage. Repeated for each cap in --caps.

  wear   Per-step temperature traces for NVIDIA default, COORD and DRLCap
         (f2 selected model) on the 10 test programs, so that thermal
         cycling and modelled wear can be compared against the default.
         Reuses drlcap_faithful code unchanged (imported, not edited).

Only frequency caps (nvidia-smi -lgc / -rgc) are changed, as in the f2
evaluation; every run ends with the cap reset.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drlcap_faithful"))

import pynvml  # noqa: E402

import agent as A  # noqa: E402
import drlcap as D  # noqa: E402
import evaluate as E  # noqa: E402
import gpu_control as G  # noqa: E402
import workloads as W  # noqa: E402

F2_MODEL = os.path.join(HERE, "..", "drlcap_faithful", "results",
                        "train_freq_s4_f2_20261113_110527", "model_final.keras")
FIELDS = ["exp", "job", "policy", "workload", "phase", "cap_mhz", "t",
          "util", "mem_util", "power", "clock", "temp", "fan", "energy_j"]


def sample(handle):
    s = G.read()
    try:
        s["fan"] = float(pynvml.nvmlDeviceGetFanSpeed(handle))
    except pynvml.NVMLError:
        s["fan"] = ""
    s["t"] = time.monotonic()
    return s


class Logger:
    """Samples NVML every 300 ms in a thread and writes tagged rows."""

    def __init__(self, writer, handle):
        self.w, self.h, self.tag = writer, handle, {}
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            row = sample(self.h)
            row.update(self.tag)
            self.w.writerow({k: row.get(k, "") for k in FIELDS})
            self.stop.wait(D.SAMPLE_INTERVAL)

    def start(self):
        self.th.start()

    def close(self):
        self.stop.set()
        self.th.join()


def wait_cool(target_c, timeout_s=900):
    t0 = time.time()
    while G.read()["temp"] > target_c and time.time() - t0 < timeout_s:
        time.sleep(5)


def exp_leak(args, log):
    burn = os.path.join(HERE, "burn")
    for rep in range(args.leak_repeats):
        for cap in args.caps:
            G.reset_cap()
            wait_cool(args.cool_c)
            log.tag = {"exp": "leak", "job": f"{rep}", "policy": "fixed",
                       "workload": "burn", "phase": "idle_pre", "cap_mhz": cap}
            time.sleep(20)
            G.set_cap(cap)
            log.tag = dict(log.tag, phase="burn")
            subprocess.run([burn, str(args.burn_s)], check=True,
                           env=W.gpu_env())
            G.reset_cap()
            log.tag = dict(log.tag, phase="cooldown")
            time.sleep(args.cool_s)
            print(f"leak rep {rep} cap {cap} done", flush=True)


def exp_wear(args, log, raw):
    path = os.path.abspath(F2_MODEL)
    variant, kind, settings, actions = E.load_model_spec(path)
    coord = {}
    for name in args.workloads:
        D.wait_gpu_free()
        coord[name] = E.profile(name)
        time.sleep(2.0)
    policies = ["baseline", "coord", "drlcap"]
    rows, job = [], 0
    drl_log = open(os.path.join(args.out, "drlcap_steps.csv"), "w", newline="")
    drl_steps = csv.DictWriter(drl_log, fieldnames=D.LOG_FIELDS)
    drl_steps.writeheader()
    try:
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                k = (rep + wi) % len(policies)
                for policy in policies[k:] + policies[:k]:
                    shared = D.wait_gpu_free()
                    log.tag = {"exp": "wear", "job": job, "policy": policy,
                               "workload": name, "phase": "run"}
                    if policy == "baseline":
                        res = E.baseline_job(name)
                    elif policy == "coord":
                        res = E.baseline_job(name, coord[name]["cap_mhz"])
                    else:
                        E.use_settings(settings)
                        ag = D.make_agent(kind, len(actions), 1000 + job)
                        ag.load(path)
                        ag.epsilon = A.EPSILON_END
                        ag.learn_start = A.LEARN_START_ONLINE
                        res = D.run_job(ag, name, actions=actions, learn=True,
                                        log=drl_steps, job_id=job)
                    log.tag = {"exp": "wear", "job": job, "policy": policy,
                               "workload": name, "phase": "gap"}
                    res.update(policy=policy, repeat=rep, job=job,
                               gpu_shared=int(bool(shared)))
                    rows.append(res)
                    with open(raw, "w", newline="") as f:
                        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                        wr.writeheader()
                        wr.writerows(rows)
                    drl_log.flush()
                    print(f"wear rep {rep} {name:12s} {policy:9s} "
                          f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J",
                          flush=True)
                    job += 1
                    time.sleep(2.0)
    finally:
        drl_log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["leak", "wear", "both"])
    ap.add_argument("--caps", type=int, nargs="+", default=[2100, 2505])
    ap.add_argument("--burn-s", type=float, default=300)
    ap.add_argument("--cool-s", type=float, default=180)
    ap.add_argument("--cool-c", type=float, default=36)
    ap.add_argument("--leak-repeats", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    args = ap.parse_args()

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(G.GPU_ID)
    args.out = os.path.join(HERE, "results",
                            f"{args.exp}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.out)
    f = open(os.path.join(args.out, "trace.csv"), "w", newline="")
    wr = csv.DictWriter(f, fieldnames=FIELDS)
    wr.writeheader()
    log = Logger(wr, handle)
    log.start()
    try:
        if args.exp in ("leak", "both"):
            exp_leak(args, log)
        if args.exp in ("wear", "both"):
            exp_wear(args, log, os.path.join(args.out, "runs.csv"))
    finally:
        log.close()
        f.close()
        G.reset_cap()
    print("saved", args.out)


if __name__ == "__main__":
    main()
