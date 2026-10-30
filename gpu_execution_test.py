import tensorflow as tf
import time

print("=" * 70)
print("TENSORFLOW GPU EXECUTION TEST")
print("=" * 70)

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

print("\nVisible GPUs:")
for gpu in tf.config.list_physical_devices("GPU"):
    print(" ", gpu)

print("\nGPU details:")
for gpu in tf.config.list_physical_devices("GPU"):
    print(tf.config.experimental.get_device_details(gpu))

# Enable TensorFlow device-placement logging
tf.debugging.set_log_device_placement(True)

print("\nStarting GPU computation...")

with tf.device("/GPU:0"):

    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])

    # Force the tensors to be materialized
    a.numpy()
    b.numpy()

    start = time.time()

    for i in range(20):
        c = tf.matmul(a, b)

        # Force synchronization
        c.numpy()

    elapsed = time.time() - start

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Result shape:", c.shape)
print("Total time:", elapsed, "seconds")
print("Average iteration:", elapsed / 20, "seconds")
print("=" * 70)
