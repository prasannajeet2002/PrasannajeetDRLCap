"""Simulation-first DDQN + PER prototype. Synthetic proxies only; no GPU controls."""
import argparse, csv, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from replay_buffer import Memory


@dataclass
class Config:
    seed: int = 7
    episodes: int = 100
    steps: int = 100
    replay_capacity: int = 10000
    batch_size: int = 64
    warmup: int = 256
    gamma: float = 0.99
    lr: float = 0.001
    target_update: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    hidden: int = 64
    # Simulated action labels only. These do not change real GPU clocks.
    clocks: tuple = (1200, 1500, 1800, 2100, 2400, 2700)


class SimulatedGPUEnv:
    """Uncalibrated synthetic environment, useful for software pipeline testing."""
    names = ("idle", "light", "compute", "memory", "mixed")

    def __init__(self, seed, steps, clocks):
        self.rng = np.random.default_rng(seed)
        self.steps, self.clocks = steps, np.asarray(clocks, np.float32)
        self.max_clock = float(max(clocks))

    def _workload(self):
        self.kind = self.rng.choice(self.names, p=[.10, .25, .30, .20, .15])
        ranges = {
            "idle": (.02, .10, .05, .20), "light": (.15, .40, .10, .40),
            "compute": (.65, 1., .10, .40), "memory": (.30, .70, .65, 1.),
            "mixed": (.50, .90, .45, .90),
        }
        a, b, c, d = ranges[self.kind]
        self.intensity = float(self.rng.uniform(a, b))
        self.mem_intensity = float(self.rng.uniform(c, d))

    def _state(self):
        return np.asarray([
            np.clip(self.util, 0, 1),
            np.clip(self.mem_intensity, 0, 1),
            np.clip(self.power / 300., 0, 1),
            np.clip((self.temp - 25.) / 75., 0, 1),
            np.clip(self.clock / self.max_clock, 0, 1),
        ], np.float32)

    def reset(self):
        self.t, self.temp = 0, 35.
        self.clock, self.power, self.util = self.max_clock, 15., 0.
        self._workload()
        return self._state()

    def step(self, action):
        self.clock = float(self.clocks[int(action)])
        if self.t % 15 == 0:
            self._workload()
        f = self.clock / self.max_clock
        compute_fraction = {"idle": .15, "light": .45, "compute": .90,
                            "memory": .20, "mixed": .60}[self.kind]
        sensitivity = .15 + .85 * compute_fraction
        performance = self.intensity * ((1 - sensitivity) * .75 + sensitivity * f)
        self.util = self.intensity * (.45 + .55 * f)
        self.power = float(np.clip(
            12 + 185 * self.util * (.25 + .75 * f**2) + 12*self.mem_intensity,
            8, 285))
        self.temp += .12 * ((28 + .22*self.power) - self.temp)
        self.temp = float(np.clip(self.temp, 28, 95))
        energy_proxy = self.power  # one simulated time unit per step
        thermal_penalty = max(0., self.temp - 75.) / 20.
        # Custom prototype reward, NOT asserted to be the paper's reward.
        reward = 2. * performance - .006 * energy_proxy - .5 * thermal_penalty
        self.t += 1
        done = self.t >= self.steps
        info = dict(
            workload=self.kind, simulated_clock_mhz=self.clock,
            gpu_util_proxy=100*self.util, memory_util_proxy=100*self.mem_intensity,
            power_w_proxy=self.power, temperature_c_proxy=self.temp,
            performance_proxy=performance, energy_j_proxy=energy_proxy)
        return self._state(), float(reward), done, info


class DDQNAgent:
    def __init__(self, cfg, n_actions):
        self.cfg, self.n_actions = cfg, n_actions
        self.memory = Memory(cfg.replay_capacity)
        self.epsilon, self.seen, self.learn_steps = cfg.epsilon_start, 0, 0

        def make_net():
            x = inp = tf.keras.Input(shape=(5,), name="state")
            x = tf.keras.layers.Dense(cfg.hidden, activation="relu")(x)
            x = tf.keras.layers.Dense(cfg.hidden, activation="relu")(x)
            return tf.keras.Model(inp, tf.keras.layers.Dense(n_actions)(x))

        self.online, self.target = make_net(), make_net()
        self.target.set_weights(self.online.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(cfg.lr)

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q = self.online(np.asarray(state, np.float32)[None, :], training=False)
        return int(tf.argmax(q[0]).numpy())

    def store(self, s, a, r, ns, done):
        # Layout: state(5), action, reward, done, next_state(5)
        tr = np.concatenate([s, np.asarray([a, r, float(done)], np.float32), ns])
        self.memory.store(tr)
        self.seen += 1
        p = min(1., self.seen / self.cfg.epsilon_decay_steps)
        self.epsilon = self.cfg.epsilon_start + p*(self.cfg.epsilon_end-self.cfg.epsilon_start)

    def learn(self):
        if self.memory.tree.size < self.cfg.batch_size:
            return None
        idx, batch, weights = self.memory.sample(self.cfg.batch_size)
        s = batch[:, :5]
        a = batch[:, 5].astype(np.int32)
        r = batch[:, 6]
        done = batch[:, 7]
        ns = batch[:, 8:13]
        s_tf, ns_tf = tf.convert_to_tensor(s), tf.convert_to_tensor(ns)

        # DDQN: online network selects next action; target network evaluates it.
        next_a = tf.argmax(self.online(ns_tf, training=False), axis=1, output_type=tf.int32)
        next_q = tf.gather(self.target(ns_tf, training=False), next_a, batch_dims=1)
        targets = tf.stop_gradient(tf.convert_to_tensor(r) +
                                   self.cfg.gamma*(1.-tf.convert_to_tensor(done))*next_q)
        w = tf.convert_to_tensor(weights[:, 0], tf.float32)
        with tf.GradientTape() as tape:
            all_q = self.online(s_tf, training=True)
            chosen_q = tf.gather(all_q, a, batch_dims=1)
            td = targets - chosen_q
            loss = tf.reduce_mean(w * tf.square(td))
        grads = tape.gradient(loss, self.online.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        abs_td = np.abs(td.numpy()).reshape(-1)
        self.memory.batch_update(idx, abs_td)
        self.learn_steps += 1

        # Diagnostic statistics for the sampled minibatch.
        self.last_td_mean = float(np.mean(abs_td))
        self.last_td_p50 = float(np.percentile(abs_td, 50))
        self.last_td_p90 = float(np.percentile(abs_td, 90))
        self.last_td_p99 = float(np.percentile(abs_td, 99))
        self.last_td_over_cap = float(np.mean(
            abs_td + self.memory.epsilon >= self.memory.abs_err_upper
        ))

        if self.learn_steps % self.cfg.target_update == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss.numpy())


def run_fixed(cfg, action, seed):
    env = SimulatedGPUEnv(seed, cfg.steps, cfg.clocks)
    s = env.reset()
    totals = dict(reward=0., energy=0., performance=0., max_temp=env.temp)
    done = False
    while not done:
        s, r, done, info = env.step(action)
        totals["reward"] += r
        totals["energy"] += info["energy_j_proxy"]
        totals["performance"] += info["performance_proxy"] / cfg.steps
        totals["max_temp"] = max(totals["max_temp"], info["temperature_c_proxy"])
    return totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    cfg = Config(episodes=args.episodes, steps=args.steps, seed=args.seed)
    random.seed(cfg.seed); np.random.seed(cfg.seed); tf.random.set_seed(cfg.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out/"config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = SimulatedGPUEnv(cfg.seed, cfg.steps, cfg.clocks)
    agent = DDQNAgent(cfg, len(cfg.clocks))
    start = time.time()
    with open(out/"training_log.csv", "w", newline="") as f:
        fields = ["episode","step","global_step","workload","action_index",
                  "simulated_clock_mhz","reward","loss","epsilon","power_w_proxy",
                  "temperature_c_proxy","performance_proxy","energy_j_proxy","done",
                  "td_mean","td_p50","td_p90","td_p99","td_over_cap"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        gstep = 0
        for ep in range(cfg.episodes):
            s, ep_reward = env.reset(), 0.
            for step in range(cfg.steps):
                a = agent.act(s)
                ns, r, done, info = env.step(a)
                agent.store(s, a, r, ns, done)
                loss = agent.learn() if agent.memory.tree.size >= max(cfg.warmup, cfg.batch_size) else None
                writer.writerow(dict(
                    episode=ep+1, step=step+1, global_step=gstep,
                    workload=info["workload"], action_index=a,
                    simulated_clock_mhz=info["simulated_clock_mhz"], reward=r,
                    loss="" if loss is None else loss, epsilon=agent.epsilon,
                    power_w_proxy=info["power_w_proxy"],
                    temperature_c_proxy=info["temperature_c_proxy"],
                    performance_proxy=info["performance_proxy"],
                    energy_j_proxy=info["energy_j_proxy"], done=done,
                    td_mean="" if loss is None else agent.last_td_mean,
                    td_p50="" if loss is None else agent.last_td_p50,
                    td_p90="" if loss is None else agent.last_td_p90,
                    td_p99="" if loss is None else agent.last_td_p99,
                    td_over_cap="" if loss is None else agent.last_td_over_cap))
                s, ep_reward, gstep = ns, ep_reward+r, gstep+1
            f.flush()
            print(f"Episode {ep+1:03d}/{cfg.episodes} reward={ep_reward:.3f} "
                  f"epsilon={agent.epsilon:.3f} replay={agent.memory.tree.size}")

    agent.online.save(out/"ddqn_per.keras")
    rows = []
    # Fixed policies and learned greedy policy evaluated on separate seeds.
    for label, action in [
        ("fixed_low", 0), ("fixed_mid", len(cfg.clocks)//2),
        ("fixed_high", len(cfg.clocks)-1)]:
        vals = [run_fixed(cfg, action, cfg.seed+1000+i) for i in range(10)]
        rows.append(dict(policy=label, **{
            k: float(np.mean([v[k] for v in vals])) for k in vals[0]
        }))
    learned = []
    for i in range(10):
        e = SimulatedGPUEnv(cfg.seed+2000+i, cfg.steps, cfg.clocks)
        s, done = e.reset(), False
        totals = dict(reward=0., energy=0., performance=0., max_temp=e.temp)
        while not done:
            a = agent.act(s, explore=False)
            s, r, done, info = e.step(a)
            totals["reward"] += r; totals["energy"] += info["energy_j_proxy"]
            totals["performance"] += info["performance_proxy"]/cfg.steps
            totals["max_temp"] = max(totals["max_temp"], info["temperature_c_proxy"])
        learned.append(totals)
    rows.append(dict(policy="learned_ddqn_per", **{
        k: float(np.mean([v[k] for v in learned]) for k in []) for k in []
    }) if False else {
        "policy":"learned_ddqn_per",
        **{k:float(np.mean([v[k] for v in learned])) for k in learned[0]}
    })
    with open(out/"evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print("\nEvaluation values are synthetic proxies, not hardware measurements:")
    for row in rows: print(row)
    print(f"Saved {out/'ddqn_per.keras'}, {out/'training_log.csv'}, {out/'evaluation.csv'}")
    print(f"Elapsed: {time.time()-start:.1f}s. No real GPU settings were changed.")


if __name__ == "__main__":
    main()
