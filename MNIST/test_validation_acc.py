import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from capsnet import CapsNet
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # set to run on CPU only

mnist = tf.keras.datasets.mnist  # stores the mnist database in a variable. these are handwritten numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # separates data set into testing and training sets

# normalize 
x_train = (x_train / 255).astype("float32").reshape([-1, 28, 28, 1])
x_test = (x_test / 255).astype("float32").reshape([-1, 28, 28, 1])
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")


capsNet = CapsNet()
reconstructor = tf.keras.models.Sequential()
reconstructor.add(tf.keras.layers.Flatten())
reconstructor.add(tf.keras.layers.Dense(512, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(1024, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(784, activation='sigmoid'))

checkpoint_path = "./MNIST_checkpoints"
ckpt = tf.train.Checkpoint(capsNet=capsNet, reconstructor=reconstructor)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
]
@tf.function(input_signature=train_step_signature)
def evaluate(inp, labels):
    digit_caps = capsNet(inp)  # (batch_size, 10, 16)
    digit_caps = digit_caps ** 2
    digit_caps = tf.reduce_sum(digit_caps, axis=-1)
    digit_caps = tf.math.sqrt(digit_caps)  # (batch_size, 10)
    loss = loss_function(labels, digit_caps)

    train_loss(loss)
    train_accuracy(labels, digit_caps)


@tf.function(input_signature=train_step_signature)
def create_reconstruction_input_capsules(inp, labels):
    digit_caps = capsNet(inp)  # (batch_size, 10, 16)
    labels = tf.one_hot(labels, depth=10)  # (batch_size, 10)
    digit_caps *= tf.reshape(labels, [-1, 10, 1])  # (batch_size, 10, 16)
    digit_caps = tf.reduce_sum(digit_caps, axis=1)  # (batch_size, 16)

    return digit_caps


def loss_function(real, pred):
    # real.shape = (batch_size,)
    # pred.shape = (batch_size, 10)
    T_k = tf.one_hot(real, 10)  # (batch_size, 10)
    mplus = .9
    mneg = .1
    lamb = 0.5
    loss = T_k * tf.math.maximum(0.0, mplus - pred) + lamb * (1 - T_k) * tf.math.maximum(0.0, pred - mneg)  # (batch_size, 10)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

for i in range(0, 10000, 100):
    print(i) if i % 1000 == 0 else ""
    evaluate(x_test[i: i + 100], y_test[i: i + 100])

print("validation results")
print("validation accuracy: ", train_accuracy.result())
print("validation loss: ", train_loss.result())







