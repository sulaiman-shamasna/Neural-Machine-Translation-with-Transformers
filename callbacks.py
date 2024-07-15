import tensorflow as tf
import datetime

def create_tensorboard_callback(log_dir="logs/"):
    log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback