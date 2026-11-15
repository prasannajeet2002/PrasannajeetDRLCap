# f2: the final faithful run (2026-11-13 to 2026-11-14)

f2 is the f1 reference configuration (`REFERENCE_CONFIG.md`) with **one
setting changed** and **one protocol step added**. Nothing of f1 was
overwritten; the f1 commands still reproduce f1 because the default minimum
cap in `drlcap.py` is still 1800 MHz.

## Why f1 was looked at again

f1 result (10 repeats): DRLCap 14.2 % energy saving, 4.0 % performance loss,
10.8 % EDP improvement, against 22 / 3 / 20 in the paper and 25.1 / 6.1 /
20.6 for COORD on the same GPU.

Measured facts from the f1 logs:

| Observation | Number |
|---|---|
| Result depends on the training seed | f1 seeds 0 / 1 / 2: 10.8 / 14.2 / 11.0 % EDP |
| A trained network uses very few caps | offline replay of the f1 training transitions: 1-4 distinct greedy caps |
| Power sensor of the 4070 Ti refreshes every 500 ms, sampling is 300 ms | 41 % of consecutive samples show the same power |
| Caps below 2100 MHz cost time and no longer save energy on compute-bound training programs | hotspot: 1670 J at 2100 MHz, 1768 J at 1800 MHz, 1806 J at 2400 MHz (`drlcap_repro/results/frequency_sweep_20261030_113800.csv`) |
| minibude gets almost nothing | 0.6 % EDP; Algorithm 1 case A forces the maximum cap (paper's design) |

Reading of these facts (a hypothesis): Eq. 1 scores the *change* in power
and clock between two samples, so holding any cap scores the same. The
network therefore settles on one "favourite" cap that depends on the seed,
and explores around it.

## The one setting changed: minimum frequency cap 1800 -> 2100 MHz

Paper, Sec. V: "we set the minimum value of the frequency cap to a higher
value supported by the power management knobs since the lower capping can
significantly negatively impact the application performance". The paper's
consumer cards use 84 % (RTX 3080 Ti, 1625-1925 MHz) and 77 % (RTX 2080 Ti,
1635-2115 MHz) of the maximum clock as the minimum. On the 4070 Ti,
2100 / 2745 = 77 %, and the training sweep above shows that caps below
2100 MHz are worse in both time and energy. f1 used 1800 MHz (66 %).

Code: `drlcap.py --cap-min 2100` (44 caps, 15 MHz apart; hybrid frequency
caps 2100, 2190, 2280, 2370, 2475, 2565, 2655, 2745). The power variant has
no frequency caps, so its f1 model is reused.

## The protocol step added: seed selection on validation programs

The paper fixes ε online to "the best-performing value found during offline
training", i.e. it tunes on the training side. In the same spirit, five
seeds (0-4) were trained and the one with the best geometric-mean EDP on the
11 validation programs (`workloads.VAL`, 3 repeats) was used. The test
programs were never used for selection (`select_model.py`,
`results/eval_val_f2_*/selected.json`). Seed 0, which was not selected, was
also run on the test programs and is reported for transparency.

Everything else is as in f1: scaled state, Algorithm 1 + Eq. 1 reward,
Table V hyper-parameters, case-A forcing, reward delay 0, a fresh copy of
the model for every test job with online learning, ε 0.1.

Other code changes (no effect on the method): `evaluate.py` reads each
model's action list from its own `config.json`, so f1 and f2 models can be
compared in one run; both scripts wait while another job uses GPU 1 and
mark a run that overlapped with one (`gpu_shared`); such runs are dropped.
None occurred in f2.

## Result (paper method, 10 test programs, 10 repeats)

| Method | Energy saved % | Perf loss % | EDP improved % |
|---|---|---|---|
| DRLCap frequency (f2, selected seed 4) | 15.0 | 3.3 | 12.2 |
| DRLCap frequency (f2, seed 0, not selected) | 12.2 | 2.8 | 9.7 |
| DRLCap power | 6.6 | 2.1 | 4.6 |
| DRLCap hybrid | 13.1 | 3.7 | 9.9 |
| COORD | 25.2 | 6.0 | 20.7 |
| RL (tabular Q-learning) | 17.0 | 4.3 | 13.5 |
| DRLCap frequency, frozen network (3 repeats) | 16.3 | 2.9 | 13.9 |
| f1 DRLCap frequency, for reference | 14.2 | 4.0 | 10.8 |
| Paper (V100S) DRLCap / COORD / RL | 22 / 25 / 5 | 3 / 5 / 1 | 20 / 21 / 4 |

Full tables, per-program numbers and timelines: `results/final_f2/RESULTS_F2.md`.

## What f2 shows

Facts:

- The change helped a little: EDP 10.8 -> 12.2 %, performance loss 4.0 ->
  3.3 % (now close to the paper's 3 %), minibude 0.6 -> 4.4 %.
- DRLCap is still well below the paper's 20 % and below COORD (20.7 %).
- The tabular RL baseline is not worse than DRLCap here (13.5 vs 12.2 %
  EDP); in the paper it is far worse (4 %).
- Seeds still differ: validation EDP of the five seeds was 3.7 to 12.1 %.
- All five seeds keep the cap in the upper part of the range (mean cap
  2490-2616 MHz on validation; the selected agent uses about 2500 MHz on
  nine of ten test programs), while COORD caps most programs at 2100-2400.
- The frozen network is slightly better than the online-learning one
  (13.9 vs 12.2 %; f1: 12.8 vs 10.8 %), so online learning does not help on
  this GPU.

Reading (hypothesis, not proven): with the paper's reward, a 300 ms sample
and a 500 ms power sensor, the agent has no signal that tells it a lower cap
is better, so it learns "one cap plus exploration noise". The gap to the
paper comes from the method on this hardware, not from a setting that was
left wrong; two faithful runs (f1, f2) and the earlier v3 run (14.9 %) all
land at 11-15 % EDP.
