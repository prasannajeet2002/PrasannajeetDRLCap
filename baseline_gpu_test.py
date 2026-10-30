import tensorflow as tf
import time

print("=" * 60)
print("RTX 4070 Ti BASELINE GPU TEST")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

# Make sure TensorFlow uses the visible GPU
with tf.device("/GPU:0"):

    # Large matrices to create sustained GPU work
    a = tf.random.normal((8192, 8192))
    b = tf.random.normal((8192, 8192))

    # Warm-up
    print("\nWarm-up...")
    for _ in range(5):
        c = tf.matmul(a, b)
        c.numpy()

    print("Warm-up completed.")

    # Actual workload
    iterations = 100

    print(f"\nRunning {iterations} matrix multiplications...")

    start = time.time()

    for i in range(iterations):
        c = tf.matmul(a, b)

        # Force TensorFlow to finish the GPU operation
        c.numpy()

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{iterations}")

    end = time.time()

elapsed = end - start

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"Total runtime : {elapsed:.3f} seconds")
print(f"Iterations    : {iterations}")
print(f"Time/iteration: {elapsed / iterations:.4f} seconds")
print("=" * 60)
