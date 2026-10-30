# -*- coding: utf-8 -*-
# 2022.8.23  –– converted to TensorFlow 2 (eager execution, Keras, tf.saved_model)
# Frequency capping in V100S
# Memory 1107MHz
# core Frequency 1597MHz ~ 135MHz  step: 7/8 MHz

import math
import os
import numpy as np
import tensorflow as tf

# Reset core frequency to default
os.system("echo JQX_ard_1234 | sudo -S nvidia-smi -rgc")

# State: (1)GPU util, (2)MEM util, (3)power, (4)temp, (5)SM-freq bucket
GPU_LABELS = ('UTIL_GPU', 'UTIL_MEM', 'POWER', 'TEMP')

MINS    = {'UTIL_GPU': 0,   'UTIL_MEM': 0,   'POWER': 30,  'TEMP': 30}
MAXS    = {'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 250, 'TEMP': 90}
BUCKETS = {'UTIL_GPU': 20,  'UTIL_MEM': 20,  'POWER': 20,  'TEMP': 30}
gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)

# GPU SM-frequency ladder (state encoding)
max_clock, min_clock, clock = 1597, 135, 1597
CLOCKS_GPU = []
while clock > min_clock:
    CLOCKS_GPU.append(clock); clock -= 7
    CLOCKS_GPU.append(clock); clock -= 8
print(CLOCKS_GPU)
clock_gpu_bucket = {CLOCKS_GPU[i]: i for i in range(len(CLOCKS_GPU))}
POWER_IN_STATE = 0

# Actions: frequency ladder from max_freq down to min_freq
gpu_limit = 1597
max_freq, min_freq, clock = 1597, 1200, 1597
GPU = []
while clock > min_freq:
    GPU.append(clock); clock -= 7
    GPU.append(clock); clock -= 8
gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}
print('action', GPU)

# RL hyper-parameters
N_FEATURES          = 5
INITIAL_EPSILON     = 0.3
FINAL_EPSILON       = 0.2
LEARNING_RATE       = 0.0001
REWARD_DECAY        = 0.95
REPLACE_TARGET_ITER = 10
MEMORY_SIZE         = 500
BATCH_SIZE          = 32
STEPS               = 2000

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(1)
tf.random.set_seed(1)
print(tf.__version__)


# ── state reader ─────────────────────────────────────────────────────────────
def state():
    os.system(
        'nvidia-smi --format=csv,noheader,nounits --filename=state.csv '
        '--query-gpu=power.draw,clocks.current.graphics,'
        'utilization.gpu,utilization.memory,temperature.gpu')
    os.system('sleep 0.3')
    with open('state.csv', 'r') as fo:
        for line in fo.readlines():
            parts     = line.replace(',', '').split()
            power_gpu = float(parts[0])
            clock_gpu = float(parts[1])
            util_gpu  = float(parts[2])
            util_mem  = float(parts[3])
            temp      = float(parts[4])
    stats = {
        'GPUL': gpu_limit, 'CLOCKS_GPU': clock_gpu,
        'UTIL_GPU': util_gpu, 'UTIL_MEM': util_mem,
        'POWER': power_gpu,  'TEMP': temp}
    print(stats)

    gpu_all_mins    = np.array([MINS[k]    for k in GPU_LABELS], dtype=np.double)
    gpu_all_maxs    = np.array([MAXS[k]    for k in GPU_LABELS], dtype=np.double)
    gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_widths      = (gpu_all_maxs - gpu_all_mins) / gpu_num_buckets

    gpu_raw   = np.clip([stats[k] for k in GPU_LABELS], gpu_all_mins, gpu_all_maxs)
    gpu_state = np.clip((gpu_raw - gpu_all_mins) / gpu_widths, 0, gpu_num_buckets - 1)
    gpu_state = np.append(gpu_state, [clock_gpu_bucket[stats['CLOCKS_GPU']]])
    if POWER_IN_STATE:
        gpu_state = np.append(gpu_state, [gpu_to_bucket[stats['GPUL']]])

    gpu_state = [int(x) for x in gpu_state]
    print("gpu_state:", gpu_state)
    return gpu_state, stats


# ── Prioritised Replay Memory ─────────────────────────────────────────────────
class SumTree:
    """Store data with its priority in the tree."""
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[cl_idx]:
                parent_idx = cl_idx
            else:
                v -= self.tree[cl_idx]
                parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory:  # stored as (s, a, r, s_) in SumTree
    epsilon                     = 0.01
    alpha                       = 0.6
    beta                        = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper               = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx     = np.empty((n,),   dtype=np.int32)
        b_memory  = np.empty((n, self.tree.data[0].size))
        ISWeights = np.empty((n, 1))
        pri_seg   = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob  = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# ── Keras network builder ─────────────────────────────────────────────────────
def build_network(n_features, n_actions, n_l1=20, trainable=True):
    """Two-layer Q-network via Keras functional API."""
    inputs = tf.keras.Input(shape=(n_features,), name='state')
    x = tf.keras.layers.Dense(
        n_l1, activation='relu', trainable=trainable,
        kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
        bias_initializer=tf.keras.initializers.Constant(0.1))(inputs)
    outputs = tf.keras.layers.Dense(
        n_actions, trainable=trainable,
        kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
        bias_initializer=tf.keras.initializers.Constant(0.1))(x)
    return tf.keras.Model(inputs, outputs)


# ── DDQN agent ────────────────────────────────────────────────────────────────
class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features=N_FEATURES,
            learning_rate=LEARNING_RATE,
            reward_decay=REWARD_DECAY,
            e_greedy=INITIAL_EPSILON,
            replace_target_iter=REPLACE_TARGET_ITER,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            output_graph=False,
            prioritized=True,
            checkpoint_path='/home/wym/project/cpu_gpu/gpu/rl/net-freq/save_net',
            savedmodel_path='/home/wym/project/cpu_gpu/gpu/rl/net-freq/model',
    ):
        self.n_actions           = n_actions
        self.n_features          = n_features
        self.lr                  = learning_rate
        self.gamma               = reward_decay
        self.epsilon             = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size         = memory_size
        self.batch_size          = batch_size
        self.prioritized         = prioritized
        self.learn_step_counter  = 0
        self.cost_his            = []

        # Paths used at save time
        self.checkpoint_path = checkpoint_path
        self.savedmodel_path = savedmodel_path

        # Build eval (trainable) and target (frozen) networks
        self.eval_net   = build_network(n_features, n_actions, trainable=True)
        self.target_net = build_network(n_features, n_actions, trainable=False)
        self._update_target_network()

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # Replay memory
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        if output_graph:
            tf.summary.create_file_writer("logs/")

    # ── internal helpers ──────────────────────────────────────────────────────
    def _update_target_network(self):
        """Hard copy: target <- eval."""
        self.target_net.set_weights(self.eval_net.get_weights())

    # ── public API ────────────────────────────────────────────────────────────
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            self.memory[self.memory_counter % self.memory_size, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        obs = observation[np.newaxis, :].astype(np.float32)
        print("epsilon:", self.epsilon)
        if np.random.uniform() > self.epsilon:
            q_values = self.eval_net(obs, training=False)
            action   = int(np.argmax(q_values.numpy()))
            print("max!!!!!!!!!!!!!!!!!!!!!")
        else:
            action = np.random.randint(0, self.n_actions)
            print("random!!!!!!!!!!!!!!!!!!!!!")
        return action

    def learn(self):
        # ── periodically copy eval -> target ─────────────────────────────────
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._update_target_network()
            print('\ntarget_params_replaced\n')

        # ── sample batch ──────────────────────────────────────────────────────
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            ISWeights = ISWeights.astype(np.float32)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        s_batch  = batch_memory[:, :self.n_features].astype(np.float32)
        s_batch_ = batch_memory[:, -self.n_features:].astype(np.float32)

        # ── DDQN target computation ───────────────────────────────────────────
        q_next      = self.target_net(s_batch_, training=False).numpy()
        q_eval4next = self.eval_net(s_batch_,   training=False).numpy()
        q_eval      = self.eval_net(s_batch,    training=False).numpy()

        q_target       = q_eval.copy()
        batch_index    = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward         = batch_memory[:, self.n_features + 1]

        # DDQN: use eval net to select action, target net to evaluate it
        max_act4next    = np.argmax(q_eval4next, axis=1)
        selected_q_next = q_next[batch_index, max_act4next]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        q_target_tf = tf.constant(q_target, dtype=tf.float32)

        # ── gradient update ───────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            q_pred = self.eval_net(s_batch, training=True)
            if self.prioritized:
                abs_errors = tf.reduce_sum(tf.abs(q_target_tf - q_pred), axis=1)
                loss = tf.reduce_mean(
                    tf.constant(ISWeights[:, 0], dtype=tf.float32)
                    * tf.reduce_sum(tf.square(q_target_tf - q_pred), axis=1))
            else:
                abs_errors = None
                loss = tf.reduce_mean(tf.square(q_target_tf - q_pred))

        grads = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.eval_net.trainable_variables))

        self.cost = float(loss.numpy())
        self.cost_his.append(self.cost)

        if self.prioritized:
            self.memory.batch_update(tree_idx, abs_errors.numpy())

        # ── epsilon decay ─────────────────────────────────────────────────────
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (STEPS - MEMORY_SIZE)
        self.epsilon  = max(self.epsilon, FINAL_EPSILON)

        self.learn_step_counter += 1

    def save(self):
        """Save model weights (TF2 SavedModel + checkpoint)."""
        # SavedModel (full model, recommended)
        self.eval_net.save(self.savedmodel_path)
        print('Save model to path:', self.savedmodel_path)

        # Checkpoint (weights only, lighter alternative)
        ckpt = tf.train.Checkpoint(eval_net=self.eval_net,
                                   target_net=self.target_net)
        save_path = ckpt.save(self.checkpoint_path)
        print('Save to path:', save_path)


# ── main ──────────────────────────────────────────────────────────────────────
RL = DQNPrioritizedReplay(n_actions=len(GPU))

observation, states = state()

for step in range(STEPS):
    action    = RL.choose_action(np.array(observation))
    gpu_limit = GPU[action]
    print('freq cap:', gpu_limit)
    os.system("echo JQX_ard_1234 | sudo -S nvidia-smi -lgc 135,%s" % gpu_limit)

    power_g = math.floor(states['POWER'])
    fre_g   = states['CLOCKS_GPU']

    observation_, states_ = state()
    power_g_ = math.floor(states_['POWER'])
    fre_g_   = states_['CLOCKS_GPU']

    power = power_g_ - power_g
    fre   = fre_g_   - fre_g
    print('power:', power, fre)

    if power <= 0:
        if   fre >= -45:              reward =  5
        elif -90  <= fre < -45:       reward = -1
        elif -135 <= fre < -90:       reward = -2
        else:                         reward = -3
    else:
        if   fre <= 45:               reward = -1
        elif 45  < fre <= 90:         reward = -2
        elif 90  < fre <= 135:        reward = -3
        else:                         reward = -4

    print('reward:', reward)
    RL.store_transition(observation, action, reward, observation_)

    if step > MEMORY_SIZE:
        RL.learn()

    states      = states_
    observation = observation_
    print('step-----------------:', step, '\n')

# ── save networks ─────────────────────────────────────────────────────────────
RL.save()

# Reset core frequency to default
os.system("echo JQX_ard_1234 | sudo -S nvidia-smi -rgc")

print('end')
