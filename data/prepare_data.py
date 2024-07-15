# data/prepare_data.py
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_text as text

def load_data(dataset_name):
    examples, metadata = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return examples['train'], examples['validation']

def load_tokenizers(model_name):
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)
    return tokenizers

def prepare_batch(tokenizers, max_tokens):
    def batch_fn(pt, en):
        pt = tokenizers.pt.tokenize(pt)[:, :max_tokens].to_tensor()
        en = tokenizers.en.tokenize(en)
        en_inputs = en[:, :-1].to_tensor()
        en_labels = en[:, 1:].to_tensor()
        return (pt, en_inputs), en_labels
    return batch_fn

def make_batches(ds, buffer_size, batch_size, batch_fn):
    return (
        ds.shuffle(buffer_size)
          .batch(batch_size)
          .map(batch_fn, tf.data.AUTOTUNE)
          .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
