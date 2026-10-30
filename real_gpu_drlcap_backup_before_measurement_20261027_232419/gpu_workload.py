import tensorflow as tf
import time
import os

print("=" * 70)
print("GPU WORKLOAD")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

# ---------------------------------------------------------
# Force TensorFlow to use the visible GPU
# ---------------------------------------------------------

with tf.device("/GPU:0"):

    # Large matrices
    size = 8192

    print("Creating matrices...")

    A = tf.random.normal([size, size])
    B = tf.random.normal([size, size])

    # Warm-up
    print("Warm-up...")

    for _ in range(5):
        C = tf.matmul(A, B)
        C.numpy()

    print("Starting workload...")
    print()

    start = time.time()

    iterations = 200

    for i in range(iterations):

        C = tf.matmul(A, B)

        # Force synchronization
        C.numpy()

        if (i + 1) % 20 == 0:
            print(
                f"Iteration {i+1}/{iterations} "
                f"Elapsed: {time.time() - start:.2f}s"
            )

    total_time = time.time() - start

print()
print("=" * 70)
print("WORKLOAD COMPLETE")
print("=" * 70)

print(f"Iterations : {iterations}")
print(f"Total time : {total_time:.3f} s")
print(f"Time/iter  : {total_time / iterations:.5f} s")
