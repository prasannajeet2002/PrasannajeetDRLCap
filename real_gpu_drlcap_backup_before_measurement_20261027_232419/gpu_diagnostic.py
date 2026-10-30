import time
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])

    # Warm up and synchronize
    _ = tf.matmul(a, b).numpy()

    print("Starting workload for approximately 30 seconds...")
    start = time.perf_counter()
    iterations = 0

    while time.perf_counter() - start < 30:
        c = tf.matmul(a, b)
        _ = c.numpy()
        iterations += 1

    elapsed = time.perf_counter() - start

print("Iterations:", iterations)
print(f"Elapsed time: {elapsed:.2f} seconds")
print("Diagnostic completed.")
