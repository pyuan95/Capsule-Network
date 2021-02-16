import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from capsnet import CapsNet
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # set to run on CPU only

mnist = tf.keras.datasets.mnist  # stores the mnist database in a variable. these are handwritten numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # separates data set into testing and training sets

# normalize the bois
x_train = (x_train / 255).astype("float32").reshape([-1, 28, 28, 1])
x_test = (x_test / 255).astype("float32").reshape([-1, 28, 28, 1])
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")

reconstructor = tf.keras.models.Sequential()
reconstructor.add(tf.keras.layers.Flatten())
reconstructor.add(tf.keras.layers.Dense(512, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(1024, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(784, activation='sigmoid'))

capsNet = CapsNet()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_reconstructor_mae = tf.keras.metrics.Mean(name='train_reconstructor_mae')
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
def train(inp, labels):
    with tf.GradientTape(persistent=True) as tape:
        digit_caps = capsNet(inp)  # (batch_size, 10, 16)

        labels_onehot = tf.one_hot(labels, depth=10)  # (batch_size, 10)
        reconstruction_vector = digit_caps * tf.reshape(labels_onehot, [-1, 10, 1])  # (batch_size, 10, 16)
        reconstructed_images = reconstructor(reconstruction_vector)  # (batch_size, 784)
        reconstruction_loss = tf.keras.losses.mean_squared_error(tf.reshape(inp, [-1, 784]), reconstructed_images)
        # reconstruction_loss *= 10
        reconstruction_mae = tf.keras.losses.mean_absolute_error(tf.reshape(inp, [-1, 784]), reconstructed_images)

        digit_caps = digit_caps ** 2
        digit_caps = tf.reduce_sum(digit_caps, axis=-1)
        digit_caps = tf.math.sqrt(digit_caps)  # (batch_size, 10)
        capsule_loss = loss_function(labels, digit_caps)

        total_loss = reconstruction_loss * 0.0005 + capsule_loss

    gradients = tape.gradient(total_loss, capsNet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, capsNet.trainable_variables))

    gradients = tape.gradient(reconstruction_loss, reconstructor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, reconstructor.trainable_variables))

    train_loss(capsule_loss)
    train_accuracy(labels, digit_caps)
    train_reconstructor_mae(reconstruction_mae)


@tf.function(input_signature=train_step_signature)
def train_reconstructor(inp, labels):
    digit_caps = capsNet(inp)  # (batch_size, 10, 16)
    with tf.GradientTape() as tape:
        labels_onehot = tf.one_hot(labels, depth=10)  # (batch_size, 10)
        reconstruction_vector = digit_caps * tf.reshape(labels_onehot, [-1, 10, 1])  # (batch_size, 10, 16)
        reconstructed_images = reconstructor(reconstruction_vector)  # (batch_size, 784)
        reconstruction_loss = tf.keras.losses.mean_squared_error(tf.reshape(inp, [-1, 784]), reconstructed_images)
        reconstruction_mae = tf.keras.losses.mean_absolute_error(tf.reshape(inp, [-1, 784]), reconstructed_images)

    gradients = tape.gradient(reconstruction_loss, reconstructor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, reconstructor.trainable_variables))

    train_reconstructor_mae(reconstruction_mae)

def loss_function(real, pred):
    # real.shape = (batch_size,)
    # pred.shape = (batch_size, 10)
    T_k = tf.one_hot(real, 10)  # (batch_size, 10)
    mplus = .9
    mneg = .1
    lamb = 0.5
    loss = T_k * (tf.math.maximum(0.0, mplus - pred) ** 2) + lamb * (1 - T_k) * (tf.math.maximum(0.0, pred - mneg) ** 2)  # (batch_size, 10)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


if __name__ == "__main__":
    batch_size = 25
    training_examples = len(x_train)
    batches = training_examples // batch_size
    print("training on ", training_examples, " examples with batch size of ", batch_size, " and ", batches, " number of batches.")

    shift_fraction = 0.1
    zoom_fraction = 0.05
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=shift_fraction,
                                                                    height_shift_range=shift_fraction)
    """
                                                                    zoom_range=zoom_fraction,
                                                                    rotation_range=15)  # shift up to 2 pixel for MNIST"""
    generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    for epoch in range(20):
        start = time.time()
        for b in range(batches):
            # train(x_train[b * batch_size: (b + 1) * batch_size], y_train[b * batch_size: (b + 1) * batch_size])
            # train_reconstructor(*generator.next())
            train(*generator.next())
            if b % 50 == 0:
                print('Epoch {} Batch {} Capsule Loss {:.4f} Reconstruction MAE {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, b, train_loss.result(), train_reconstructor_mae.result(), train_accuracy.result()))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        if (epoch + 1) % 5 == 0 or True:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        loss_log = open("capsNetMNIST_loss.txt", "a+")
        loss_log.write('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        loss_log.write("\n")
        loss_log.close()

        train_accuracy.reset_states()
        train_loss.reset_states()
        train_reconstructor_mae.reset_states()

    reconstructor.save("reconstructor.h5")

# print("primary caps equality? ", not False in tf.math.equal(primary_caps, primary_caps_test).numpy())
# print("first capsule from backup primary caps: ", primary_caps_backup[0, 0, 0, 0, 40:48])
# print("first capsule from primary caps: ", primary_caps[0, 5, :])
# print("first capsule from primary caps test: ", primary_caps_test[0, 5, :])






