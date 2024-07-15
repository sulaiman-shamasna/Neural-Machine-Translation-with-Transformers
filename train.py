# train.py
import tensorflow as tf
from models.transformer import Transformer
from utils.learning_rate import CustomSchedule
from callbacks import create_tensorboard_callback

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, tf.int64)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

def train_model(dataset, val_dataset, config):
    learning_rate = CustomSchedule(config['d_model'], warmup_steps=config['learning_rate_warmup_steps'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = Transformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        dff=config['dff'],
        input_vocab_size=config['input_vocab_size'],
        target_vocab_size=config['target_vocab_size'],
        dropout_rate=config['dropout_rate']
    )

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )

    tensorboard_callback = create_tensorboard_callback()

    transformer.fit(
        dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=[tensorboard_callback]
    )
    return transformer
