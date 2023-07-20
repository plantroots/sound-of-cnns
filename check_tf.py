import tensorflow as tf

if __name__ == "__main__":
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
