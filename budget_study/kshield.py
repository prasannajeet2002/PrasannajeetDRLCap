"""Kernel-timing slowdown estimate for the budget shield (progress-signal ladder step 3).

Offline (kernel_test.py, kernel_fit_test.py on ksweep_20261123_112830) the
slowdown measured from per-kernel durations had 0.43-0.61 pp mean error in
the 2-8 % band against 1.06 pp for E2, at +0.4 % tracer overhead. This
module does the same online.

cupti/libkernel_stream.so streams, per CUDA process, kernel counts and summed
durations per kernel name and 50 ms wall-clock bin. KernelBudget extends
shield.Budget:

  * each bin whose kernels run inside one 300 ms step (after a 30 ms settling
    time for the new cap) is tagged with the SM clock read at the end of that
    step, rounded to 30 MHz;
  * reference duration of each kernel at the default clock:
      - the mean over clock levels >= REF_CLOCK (the max cap runs at about
        the default clock), when seen there at least MIN_COUNT times;
      - otherwise d = a + c / f fitted from the two highest levels in
        [FIT_MIN, REF_CLOCK) at least 90 MHz apart, extrapolated to F_REF
        (fits that use deep caps broke on GROMACS offline, hence FIT_MIN);
  * added time = sum over all kernels of (duration - reference); work at
    default speed W = T - added, slowdown = T / W - 1, exactly as in Budget;
  * the kernel estimate is used only while it is reliable: at least 0.5 s
    of kernel time, references for >= 90 % of it, and <= 10 % of it from
    kernels that ran faster at a lower clock (the cuDNN programs offline).
    Otherwise the E2 estimate of Budget is used.
The one-step look-ahead s(c) of a candidate cap still comes from E2.
"""

import bisect
import glob
import os
import time
from collections import defaultdict

import numpy as np

from shield import F_REF, Budget

LEVEL = 30.0          # MHz, clock level resolution
REF_CLOCK = 2670.0    # levels at or above this count as the default clock
FIT_MIN = 2300.0      # lowest level used for the a + c/f fit
MIN_COUNT = 3
TOL = 0.98            # a lower-clock mean below TOL * reference flags the kernel
BIN = 0.05            # s, matches BIN_NS in kernel_stream.c
SETTLE = 0.03         # s after a step's start before a bin is tagged


class KernelStream:
    """Incremental reader of the *.stream files in one directory."""

    def __init__(self, path):
        self.path = path
        self.pos = {}
        self.names = {}                     # (file, id) -> kernel name

    def poll(self):
        """New (bin start s, name, count, sum s) records since the last poll."""
        out = []
        for p in glob.glob(os.path.join(self.path, "*.stream")):
            try:
                with open(p, "rb") as f:
                    f.seek(self.pos.get(p, 0))
                    data = f.read()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1
            self.pos[p] = self.pos.get(p, 0) + end
            for line in data[:end].decode(errors="replace").splitlines():
                if line.startswith("N,"):
                    _, i, name = line.split(",", 2)
                    self.names[(p, i)] = name
                elif line.startswith("K,"):
                    _, b, i, n, s = line.split(",")
                    out.append((int(b) / 1000.0, self.names.get((p, i), "?"),
                                int(n), int(s) / 1e9))
        return out


class KernelBudget(Budget):
    def __init__(self, ktrace_dir, delta_pct=5.0, margin_pct=1.0):
        super().__init__(delta_pct, margin_pct)
        self.stream = KernelStream(ktrace_dir)
        self.total = defaultdict(lambda: [0, 0.0])                       # name -> count, sum
        self.lev = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))    # name -> level -> count, sum
        self.steps_t, self.steps_f = [], []   # wall start of each step, its level
        self.pending = []
        self.wall_last = None
        self.k_valid = 0                     # steps that used the kernel estimate
        self.n_obs = 0
        self.k_slow = float("nan")
        self.e2_slow = 0.0
        self._W_k = None
        self.diag = []                        # one row per step, see save_diag
        self._info = (0.0, 0.0, 0.0, 0.0)

    def save_diag(self, path):
        with open(path, "w") as f:
            f.write("wall,clock,T,e2_slow,k_slow,valid,kernel_s,covered_s,unstable_s,added_s\n")
            for r in self.diag:
                f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in r) + "\n")

    # ---- data ---------------------------------------------------------
    def observe(self, util, mem, clock, now=None):
        super().observe(util, mem, clock, now)
        wall = time.time()
        if self.wall_last is not None:
            self.steps_t.append(self.wall_last)
            self.steps_f.append(LEVEL * round(clock / LEVEL))
        self.wall_last = wall
        recs = self.pending + self.stream.poll()
        self.pending = []
        for b, name, n, s in recs:
            if b + BIN > wall:                 # its step has not ended yet
                self.pending.append((b, name, n, s))
                continue
            i = bisect.bisect_right(self.steps_t, b) - 1
            t = self.total[name]
            t[0] += n
            t[1] += s
            if i < 0:
                continue
            start = self.steps_t[i]
            end = self.steps_t[i + 1] if i + 1 < len(self.steps_t) else wall
            # the bin's kernels must also finish inside the step (long kernels
            # such as miniBUDE's 127 ms ones would straddle a cap change)
            if b >= start + SETTLE and b + BIN + s / n <= end:
                lv = self.lev[name][self.steps_f[i]]
                lv[0] += n
                lv[1] += s
        self.n_obs += 1
        self._update()
        self.diag.append((wall, clock, self.T, self.e2_slow,
                          self.k_slow if self._W_k is not None else float("nan"),
                          int(self._W_k is not None), *self._info))

    # ---- estimate -----------------------------------------------------
    def _reference(self, name):
        """(reference duration or None, unstable)."""
        levels = {f: v for f, v in self.lev[name].items() if v[0] >= MIN_COUNT}
        hi = [f for f in levels if f >= REF_CLOCK]
        if hi:
            n = sum(levels[f][0] for f in hi)
            ref = sum(levels[f][1] for f in hi) / n
            low = [levels[f][1] / levels[f][0] for f in levels if f < REF_CLOCK]
            return ref, any(d < TOL * ref for d in low)
        cand = sorted((f for f in levels if FIT_MIN <= f < REF_CLOCK), reverse=True)
        if not cand:
            return None, False
        fa = cand[0]
        fb = next((f for f in cand[1:] if fa - f >= 90.0), None)
        if fb is None:
            return None, False
        da = levels[fa][1] / levels[fa][0]
        db = levels[fb][1] / levels[fb][0]
        c = (db - da) / (1.0 / fb - 1.0 / fa)
        if c < 0:
            return da, True
        a = max(da - c / fa, 0.0)
        return a + c / F_REF, False

    def _update(self):
        self.e2_slow = super().slowdown()
        tot = sum(s for _, s in self.total.values())
        self._W_k = None
        self._info = (tot, 0.0, 0.0, 0.0)
        if tot < 0.5 or self.T <= 0:
            return
        covered = unstable = added = 0.0
        for name, (n, s) in self.total.items():
            ref, bad = self._reference(name)
            if ref is None:
                continue
            covered += s
            if bad:
                unstable += s
            added += s - n * ref
        self._info = (tot, covered, unstable, added)
        if covered < 0.9 * tot or unstable > 0.1 * tot:
            return
        W = self.T - max(added, 0.0)
        if W > 0:
            self._W_k = W
            self.k_valid += 1
            self.k_slow = self.T / W - 1.0

    def slowdown(self):
        return self.e2_slow if self._W_k is None else self.k_slow

    def allowed(self, caps):
        if self._W_k is None:
            return super().allowed(caps)
        W_e2 = self.W
        self.W = self._W_k              # same look-ahead, kernel-based work
        try:
            return super().allowed(caps)
        finally:
            self.W = W_e2
