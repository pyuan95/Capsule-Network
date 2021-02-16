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
x_train = (x_train / 255).astype("float32")
x_test = (x_test / 255).astype("float32")
y_train = y_train.astype("int32")
y_test = y_test.astype("int32")

capsNet = CapsNet()
reconstructor = tf.keras.models.Sequential()
reconstructor.add(tf.keras.layers.Flatten())
reconstructor.add(tf.keras.layers.Dense(512, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(1024, activation='relu'))
reconstructor.add(tf.keras.layers.Dense(784, activation='sigmoid'))

# load most recent CapsNet
checkpoint_path = "./MNIST_checkpoints"
ckpt = tf.train.Checkpoint(capsNet=capsNet, reconstructor=reconstructor)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

def process_image(img):
    img = img.reshape([1, 28, 28, 1])
    digit_caps = capsNet(img)[0] # (10, 16)
    magnitudes = tf.math.sqrt(tf.reduce_sum(digit_caps ** 2, axis=-1))
    values, indices = tf.math.top_k(magnitudes, k=2)
    predicted_number, second_number = indices
    predicted_number_vector = digit_caps[predicted_number]
    predicted_number_chance, second_number_chance = values
    reconstructed_image = get_reconstructed_image(predicted_number_vector, predicted_number)

    return predicted_number.numpy(), predicted_number_chance.numpy(), second_number.numpy(), \
           second_number_chance.numpy(), predicted_number_vector.numpy(), reconstructed_image


def get_reconstructed_image(vector, predicted_number):
    v = np.zeros([1, 160])
    v[0, predicted_number * 16: (predicted_number + 1) * 16] = vector
    return reconstructor.predict(v).reshape([28,28])


if __name__ == "__main__":
    img = x_test[256]
    predicted_number, predicted_number_chance, predicted_number_vector, reconstructed_image = process_image(img)
    print(predicted_number, predicted_number_chance)
    plt.imshow(reconstructed_image)
    plt.show()
    plt.imshow(img)
    plt.show()

# reconstructed_image = create_reconstruction_input_capsules(x_test[:100], y_test[:100])
# reconstructed_image = reconstructor.predict(reconstructed_image)
# reconstructed_image = reconstructed_image.reshape([-1, 28, 28])
#
# index = 1
# plt.imshow(reconstructed_image[index])
# plt.show()
#
# plt.imshow(x_test[index])
# plt.show()



