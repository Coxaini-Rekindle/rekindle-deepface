"""GPU configuration utilities for deep learning models."""

import os

import tensorflow as tf


class GPUManager:
    """Handles GPU configuration and optimization for TensorFlow."""

    @staticmethod
    def configure_gpu():
        """Configure TensorFlow to use GPU efficiently.

        Returns:
            bool: True if GPU was successfully configured, False otherwise
        """
        print("Configuring GPU settings...")

        # Check for CUDA environment variables
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")

        # Check if GPU is available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Print GPU info first
                print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s):")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")

                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  Memory growth enabled for {gpu.name}")

                # Set visible devices and log GPU info
                tf.config.set_visible_devices(gpus, "GPU")
                print("GPU devices set as visible to TensorFlow")

                # Optional: Set TensorFlow to use mixed precision for faster computation
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision policy enabled")

                return True
            except RuntimeError as e:
                print(f"GPU configuration error: {str(e)}")
                return False
        else:
            print("No GPU found. Using CPU.")
            return False
