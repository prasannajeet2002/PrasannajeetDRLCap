"""DRLCap heuristic reward (paper Algorithm 1 and Eq. 1).

Inputs are two consecutive telemetry samples (before and after the action)
and the action's cap. The four cases follow the paper:

  A  u_mem == 0 and u_core == 100  fully compute-bound: the cap is raised
                                   to max and the reward bins the power
                                   change dp into r1 > r2 > r3 > r4.
  B  u_mem == 0 and u_core == 0    idle: r = c / cap, so lower caps win.
  C  u_mem == 0, otherwise         r = g / u_core
  D  u_mem > 0                     r = g * u_mem

g (Eq. 1) bins the actual clock change df in multiples of n*step. When
power did not rise (dp <= 0) scores g1 > g2 > g3 > g4 are positive and a
smaller frequency drop scores higher (power saved without giving up
speed). When power rose (dp > 0) the scores are negative ("low score"),
and again a smaller df scores higher.

The paper does not publish the numeric values of r1..r4, g1..g4, p1..p3,
n or c. g1..g4 (both branches) and the 45 MHz bin width come from the
authors' own V100S scripts kept in the project root
(Freq_DDQN_prioritized_save_tf2.py, a TF2 port of their code): there the
reward is +5/-1/-2/-3 when power did not rise and -1/-2/-3/-4 when it rose,
for frequency changes binned at 45/90/135 MHz, with power floored to whole
watts before differencing. Those scripts contain only Eq. 1, not the
Algorithm 1 cases, so r1..r4, p1..p3 and c remain OUR choices, scaled so
the best case-A and case-B rewards equal g1.
Utilisations are used as fractions in [0, 1] (NVML reports integer %).
"""

import math

# Eq. 1 frequency bin width: n * step. The authors' V100S code bins at
# 45 MHz (6 x 7.5 MHz steps); we keep 45 MHz = 3 x the 15 MHz 4070 Ti step.
STEP_MHZ = 15
N = 3

# Scores when power did not increase (g1 > g2 > g3 > g4), authors' code.
G_POWER_DOWN = (5.0, -1.0, -2.0, -3.0)
# Scores when power increased ("low score"; smaller df still better).
G_POWER_UP = (-1.0, -2.0, -3.0, -4.0)

# Case A power-change thresholds in watts (p1 < p2 < p3) and rewards
# (our choice, r1 = g1).
P_THRESH_W = (5.0, 15.0, 30.0)
R_CASE_A = (5.0, 2.5, 0.0, -2.5)

# Case B: r = c / cap with c = R_IDLE_MAX * lowest cap, so the idle reward
# lies in (0, g1] and is largest for the lowest cap (our choice).
R_IDLE_MAX = 5.0

# Case C divides by u_core, so a 1% utilisation sample would multiply g by
# 100 and swamp the Q-network's MSE loss. DEVIATION from the paper (which
# does not mention it): rewards are clipped to +-REWARD_CLIP.
REWARD_CLIP = 10.0


def _bin(x, edges):
    """Index of the first edge x is <= to, else len(edges)."""
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def score_g(dp, df):
    """Eq. 1: piecewise score from power change dp (W) and clock change df."""
    w = N * STEP_MHZ
    if dp <= 0:
        # -w <= df -> g1, -2w <= df < -w -> g2, -3w <= df < -2w -> g3, else g4
        if df >= -w:
            return G_POWER_DOWN[0]
        if df >= -2 * w:
            return G_POWER_DOWN[1]
        if df >= -3 * w:
            return G_POWER_DOWN[2]
        return G_POWER_DOWN[3]
    # power rose: the smaller the frequency rise, the better
    return G_POWER_UP[_bin(df, (w, 2 * w, 3 * w))]


def case_of(sample):
    """Which Algorithm 1 case a telemetry sample falls into."""
    if sample["mem_util"] == 0:
        if sample["util"] == 100:
            return "A"
        if sample["util"] == 0:
            return "B"
        return "C"
    return "D"


def reward(prev, cur, cap_mhz, c_idle):
    """Reward for the action that moved the GPU from `prev` to `cur`."""
    # whole watts, as in the authors' code, so sensor noise below 1 W
    # does not flip the sign of dp
    dp = math.floor(cur["power"]) - math.floor(prev["power"])
    df = cur["clock"] - prev["clock"]
    case = case_of(cur)
    if case == "A":
        return R_CASE_A[_bin(dp, P_THRESH_W)], case
    if case == "B":
        return R_IDLE_MAX * c_idle / cap_mhz, case
    g = score_g(dp, df)
    if case == "C":
        r = g / (cur["util"] / 100.0)
    else:
        r = g * (cur["mem_util"] / 100.0)
    return max(-REWARD_CLIP, min(REWARD_CLIP, r)), case
