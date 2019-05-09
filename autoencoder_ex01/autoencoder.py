import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.layers import Activation, Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard



# MNIST DATA
(x_train, _), (x_test, _) = mnist.load_data()
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parameters
loss_function = "mse"
filtersizes = [256, 128]
latentsize = 128
batchsize = 256
epoch = 20
padding = "same"
activation = "relut"

# Autoencoder
# input shape: ( 28, 28, 1)
input_shape = (image_size, image_size, 1)

# Encoder
input_img = Input(shape=input_shape)
x = input_img
x = Conv2D(filters=filtersizes[0],
           kernel_size=(3,3),
           strides=(2, 2),
           padding=padding,
           activation=activation,
           use_bias=True,
           kernel_initializer='random_normal',
           bias_initializer='random_normal'
           )(x)
x = Conv2D(filters=filtersizes[1],
           kernel_size=(3,3),
           strides=(2, 2),
           padding=padding,
           activation=activation,
           use_bias=True,
           kernel_initializer='random_normal',
           bias_initializer='random_normal'
           )(x)

shape = K.int_shape(x)


latent_vector = Dense(latentsize, activation=activation,
           use_bias=True,
           kernel_initializer='random_normal',
           bias_initializer='random_normal'
           )(x)

# Decoder

# ## using transpose convolution lead to checkerboard artifacts as results
# x = Conv2DTranspose(filters=filtersizes[0],
#            kernel_size=(3,3),
#            strides=(2, 2),
#            padding="same",
#            activation="relu",
#            use_bias=True,
#            kernel_initializer='random_normal',
#            bias_initializer='random_normal'
#            )(latent_vector)
# x = Conv2DTranspose(filters=filtersizes[1],
#            kernel_size=(3,3),
#            strides=(2, 2),
#            padding="same",
#            activation="relu",
#            use_bias=True,
#            kernel_initializer='random_normal',
#            bias_initializer='random_normal'
#            )(x)

# instead: nearest neighbour upsampling layers
x = UpSampling2D((2,2))(latent_vector)
x = Conv2D(filters=filtersizes[0],
           kernel_size=(3,3),
           padding=padding,
           activation=activation,
           use_bias=True,
           kernel_initializer='random_normal',
           bias_initializer='random_normal'
           )(x)
x = UpSampling2D((2,2))(x)
output_img = Conv2D(1,
           kernel_size=(3,3),
           padding=padding,
           activation=activation,
           use_bias=True,
           kernel_initializer='random_normal',
           bias_initializer='random_normal'
           )(x)


autoencoder = Model(input_img, output_img)
autoencoder.summary()

# Training
autoencoder.compile(optimizer="sgd",loss=loss_function,metrics=["mae","binary_crossentropy"])
autoencoder.fit(x_train, x_train,
          epochs=epoch, #epoch
          batch_size=batchsize,
          callbacks=[TensorBoard(log_dir='./summary')])
# Saving
autoencoder.save('autoencoder.h5')
# Prediction
reconstructed = autoencoder.predict(x_test)

# Plotting
plt.figure(figsize=(40, 4))
for i in range(10):
    # original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed images
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
plt.savefig('autoencoder_'+ loss_function +'.png')