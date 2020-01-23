# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:52:25 2019

@author: Sambandamn
"""


import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import math
from pprint import pprint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam, SGD


#%%

# reparameterization trick
# sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
#%%    
def plotLoss(train, test, title):
    plt.xlabel("Number of Iterations(epoch)")
    plt.ylabel("Loss")
    plt.title(title)
    itera = list(range(len(train)))
    
    plt.plot(itera, train, label="train")
    plt.plot(itera, test, label="test")       

   
    plt.grid(True, linestyle='dotted')
    plt.legend() #loc="upper left"

    plt.show()

#%%
# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print( x_train.shape)
print( x_test.shape)

#%%
# network parameters
input_shape = (original_dim,)
intermediate_dim = 256
batch_size = 128
latent_dim = 2
epochs = 50


# VAE model = encoder + decoder
# build the encoder model
inputs = Input(shape=input_shape)
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

#%%
# reparameterization trick 
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#%%

# build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate the decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate the VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

 # VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.summary()

# train the autoencoder
vae_hist=vae.fit(x_train,epochs=epochs,batch_size=batch_size, validation_data=(x_test, None))
#%%
vae_hist = vae_hist.history
## save history
#with open('./experiment/vae.hist', 'w', encoding="utf-8") as fout:
#    pprint(vae_hist, fout)
  
# show loss vs iteration graph
plotLoss(vae_hist['loss'], vae_hist['val_loss'],'Loss vs.Epoch Latent Dim = %d, Mid Dim = %d' % (latent_dim, intermediate_dim))
#plt.savefig('images/VAE_MNIST_loss_latent%d_intermediate%d.png' % (latent_dim, intermediate_dim))

x_test_encoded = vae.predict(x_test, batch_size=batch_size)


#%%
# Plot of original image on the top and the generated images on the bottom
np.random.seed(42)
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images = []

for digit in digits:
    index0 = np.where(y_test==digit)[0]
    pick = index0[np.random.choice(index0.shape[0])]
    #digit_indexes.append(pick)
    #print(pick)

    original = x_test[pick].reshape(image_size, image_size)
    encoded = x_test_encoded[pick].reshape(image_size, image_size)
    images.append([original, encoded])
    
plt.figure(figsize=(12, 4))
for i, entry in enumerate(images):
    ax1 = plt.subplot(2,10,i+1)
    ax1.imshow(entry[0], cmap=plt.cm.gray)
    ax2 = plt.subplot(2,10,i+11)
    ax2.imshow(entry[1], cmap=plt.cm.gray)


#plt.savefig('images/VAE_MNIST_gallery_latent%d_intermediate%d.png' % (latent_dim, intermediate_dim))


