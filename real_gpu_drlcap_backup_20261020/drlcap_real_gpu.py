import os
import time
import random
import subprocess
import csv
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

GPU_ID = 1

STATE_SIZE = 5

# We'll start with a small action space.
# These are graphics-clock values supported by the 4070 Ti.
ACTIONS = [
    1200,
    1500,
    1800,
    2100,
    2400,
    2700
]

NUM_ACTIONS = len(ACTIONS)

GAMMA = 0.95

LEARNING_RATE = 0.001

EPSILON_START = 0.30

EPSILON_MIN = 0.10

EPSILON_DECAY = 0.995

REPLAY_SIZE = 512

BATCH_SIZE = 32

TARGET_UPDATE = 8

TRAINING_STEPS = 500


# ============================================================
# GPU TELEMETRY
# ============================================================

def get_gpu_state():

    command = [
        "nvidia-smi",
        "-i", str(GPU_ID),
        "--query-gpu="
        "utilization.gpu,"
        "utilization.memory,"
        "power.draw,"
        "clocks.gr,"
        "temperature.gpu",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    values = [
        float(x.strip())
        for x in result.stdout.strip().split(",")
    ]

    gpu_util = values[0]
    mem_util = values[1]
    power = values[2]
    clock = values[3]
    temp = values[4]

    return np.array([
        gpu_util / 100.0,
        mem_util / 100.0,
        power / 300.0,
        clock / 3105.0,
        temp / 100.0
    ], dtype=np.float32)


# ============================================================
# SET GPU CLOCK
# ============================================================

def set_clock(clock):

    command = [
        "sudo",
        "nvidia-smi",
        "-i", str(GPU_ID),
        "-lgc", str(clock)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("Clock change failed:")
        print(result.stderr)

    else:

        print(f"GPU clock set to {clock} MHz")


# ============================================================
# RELEASE CLOCK
# ============================================================

def release_clock():

    subprocess.run(
        [
            "sudo",
            "nvidia-smi",
            "-i", str(GPU_ID),
            "-rgc"
        ]
    )

    print("GPU clock lock released.")


# ============================================================
# DDQN NETWORK
# ============================================================

def create_network():

    model = Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(STATE_SIZE,)
        ),

        Dense(
            128,
            activation="relu"
        ),

        Dense(
            NUM_ACTIONS,
            activation="linear"
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse"
    )

    return model


# ============================================================
# AGENT
# ============================================================

class DDQNAgent:

    def __init__(self):

        self.online_network = create_network()

        self.target_network = create_network()

        self.target_network.set_weights(
            self.online_network.get_weights()
        )

        self.memory = deque(
            maxlen=REPLAY_SIZE
        )

        self.epsilon = EPSILON_START

        self.training_step = 0


    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randrange(NUM_ACTIONS)

        q_values = self.online_network.predict(
            state.reshape(1, -1),
            verbose=0
        )

        return int(
            np.argmax(q_values[0])
        )


    def train(self):

        if len(self.memory) < BATCH_SIZE:

            return

        batch = random.sample(
            self.memory,
            BATCH_SIZE
        )

        states = np.array(
            [x[0] for x in batch]
        )

        actions = np.array(
            [x[1] for x in batch]
        )

        rewards = np.array(
            [x[2] for x in batch]
        )

        next_states = np.array(
            [x[3] for x in batch]
        )

        dones = np.array(
            [x[4] for x in batch]
        )

        current_q = self.online_network.predict(
            states,
            verbose=0
        )

        # ----------------------------------------------------
        # DDQN:
        #
        # online network selects action
        # target network evaluates action
        # ----------------------------------------------------

        next_online_q = self.online_network.predict(
            next_states,
            verbose=0
        )

        next_actions = np.argmax(
            next_online_q,
            axis=1
        )

        next_target_q = self.target_network.predict(
            next_states,
            verbose=0
        )

        target_values = (
            rewards
            +
            GAMMA
            *
            (
                1.0 - dones
            )
            *
            next_target_q[
                np.arange(BATCH_SIZE),
                next_actions
            ]
        )

        current_q[
            np.arange(BATCH_SIZE),
            actions
        ] = target_values

        self.online_network.fit(
            states,
            current_q,
            epochs=1,
            verbose=0
        )

        self.training_step += 1

        if self.training_step % TARGET_UPDATE == 0:

            self.target_network.set_weights(
                self.online_network.get_weights()
            )

        if self.epsilon > EPSILON_MIN:

            self.epsilon *= EPSILON_DECAY


# ============================================================
# REWARD
# ============================================================

def calculate_reward(
    old_state,
    new_state
):

    old_power = old_state[2]

    new_power = new_state[2]

    old_clock = old_state[3]

    new_clock = new_state[3]

    old_util = old_state[0]

    new_util = new_state[0]

    # --------------------------------------------------------
    # Reward:
    #
    # Reward useful GPU work.
    # Penalize power.
    # Penalize unnecessary high frequency.
    # --------------------------------------------------------

    performance = new_util

    energy_penalty = new_power

    frequency_penalty = new_clock

    reward = (
        3.0 * performance
        -
        1.2 * energy_penalty
        -
        0.5 * frequency_penalty
    )

    return float(
        np.clip(
            reward,
            -2.0,
            2.0
        )
    )


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "drlcap_results.csv"


with open(
    LOG_FILE,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "step",
        "action",
        "clock_mhz",
        "gpu_util",
        "memory_util",
        "power_w",
        "temperature_c",
        "reward",
        "cumulative_reward",
        "epsilon"
    ])


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

agent = DDQNAgent()

cumulative_reward = 0.0

print("=" * 70)
print("REAL GPU DRLCAP EXPERIMENT")
print("=" * 70)

print("GPU:", GPU_ID)

print("Actions:", ACTIONS)

print("Training steps:", TRAINING_STEPS)

print()


try:

    for step in range(
        TRAINING_STEPS
    ):

        # ----------------------------------------------
        # Observe
        # ----------------------------------------------

        state = get_gpu_state()

        # ----------------------------------------------
        # Choose action
        # ----------------------------------------------

        action_index = agent.choose_action(
            state
        )

        selected_clock = ACTIONS[
            action_index
        ]

        print(
            f"\nStep {step+1}/{TRAINING_STEPS}"
        )

        print(
            f"Action: {selected_clock} MHz"
        )

        # ----------------------------------------------
        # Apply frequency
        # ----------------------------------------------

        set_clock(
            selected_clock
        )

        # Give GPU time to react
        time.sleep(0.5)

        # ----------------------------------------------
        # Let workload run
        # ----------------------------------------------

        workload_start = time.time()

        # Short GPU computation
        with tf.device("/GPU:0"):

            a = tf.random.normal(
                [4096, 4096]
            )

            b = tf.random.normal(
                [4096, 4096]
            )

            for _ in range(3):

                c = tf.matmul(
                    a,
                    b
                )

                c.numpy()

        workload_time = (
            time.time()
            -
            workload_start
        )

        # ----------------------------------------------
        # Observe new state
        # ----------------------------------------------

        next_state = get_gpu_state()

        # ----------------------------------------------
        # Reward
        # ----------------------------------------------

        reward = calculate_reward(
            state,
            next_state
        )

        cumulative_reward += reward

        done = False

        # ----------------------------------------------
        # Store experience
        # ----------------------------------------------

        agent.remember(
            state,
            action_index,
            reward,
            next_state,
            done
        )

        # ----------------------------------------------
        # Train DDQN
        # ----------------------------------------------

        agent.train()

        # ----------------------------------------------
        # Log
        # ----------------------------------------------

        raw_gpu_util = next_state[0] * 100

        raw_memory_util = next_state[1] * 100

        raw_power = next_state[2] * 300

        raw_clock = next_state[3] * 3105

        raw_temp = next_state[4] * 100

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                step + 1,
                action_index,
                selected_clock,
                raw_gpu_util,
                raw_memory_util,
                raw_power,
                raw_temp,
                reward,
                cumulative_reward,
                agent.epsilon
            ])

        print(
            f"GPU utilization : {raw_gpu_util:.1f}%"
        )

        print(
            f"Memory utilization : {raw_memory_util:.1f}%"
        )

        print(
            f"Power : {raw_power:.2f} W"
        )

        print(
            f"Temperature : {raw_temp:.1f} C"
        )

        print(
            f"Reward : {reward:.4f}"
        )

        print(
            f"Cumulative reward : {cumulative_reward:.4f}"
        )

        print(
            f"Epsilon : {agent.epsilon:.4f}"
        )

        print(
            f"Workload time : {workload_time:.3f}s"
        )


except KeyboardInterrupt:

    print(
        "\nTraining interrupted."
    )


finally:

    # VERY IMPORTANT
    release_clock()

    print(
        "GPU restored to automatic clock control."
    )


# ============================================================
# SAVE MODEL
# ============================================================

agent.online_network.save(
    "drlcap_model.keras"
)

print(
    "\nModel saved as drlcap_model.keras"
)

print(
    "Results saved as drlcap_results.csv"
)
