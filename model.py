
# coding: utf-8

# In[19]:


from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, AveragePooling2D,Conv2DTranspose,Conv2D
from keras.optimizers import SGD
from keras import backend as K

import numpy as np
from PIL import Image, ImageOps
import argparse
import math
import os
import os.path
import glob
        


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers


# In[3]:


#K.set_image_dim_ordering('th')  # ensure our dimension notation matches


# In[22]:


def generator_model():
    model = Sequential()
    #Adding a fully connected layer and reshape the vector
    model.add(Dense(1024*4*4, input_dim=100, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(Activation('relu'))
    model.add(Reshape((4,4,1024)))
    
     # Transposed convolution layer, from 4x4x1024 into 8x8x512 tensor
    model.add(Conv2DTranspose(512, kernel_size = 3, strides = 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

              
     # Transposed convolution layer, from 8x8x512  to 16x16x256 tensor
    model.add(Conv2DTranspose(256, kernel_size = 3, strides = 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

     # Transposed convolution layer, from 16x16x256 to 32x32x128 tensor

    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Transposed convolution layer, from 32x32x128 to 64x64x3  tensor

    model.add(Conv2DTranspose(3, kernel_size = 3, strides = 2, padding='same'))
    model.add(Activation('tanh'))
    model.summary()
    return model


# In[26]:


def discriminator_model():
    depth=64
    model = Sequential()
    model.add(Conv2D(depth, kernel_size=3, strides=2, input_shape=(64,64,3),padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    #suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.
    model.add(Conv2D(depth*2, kernel_size=3, strides=2,padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(depth*4, kernel_size=3, strides=2,padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(0.25))

    model.add(Conv2D(depth*8, kernel_size=3, strides=2,padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(0.25))
    
 #For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


# In[25]:


model = discriminator_model()
print("-- Discriminator -- ")
model.summary()


# In[23]:


model = generator_model()
print("-- Generator -- ")
model.summary()


# In[6]:


# We will use the Adam optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


# In[28]:


def get_discriminator():
    optimizer = get_optimizer()
    model= Sequential()
    model.add(discriminator_model())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[29]:


def get_generator():
    optimizer = get_optimizer()
    model= Sequential()
    model.add(generator_model())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[30]:


def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


# In[35]:


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


# In[ ]:


def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train, y_train, x_test, y_test = ////
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator()
    discriminator = get_discriminator()
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        #For each epoch, we use tqdm to make our loops show a smart progress meter.
        for _ in tqdm(range(int(batch_count))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            #We take the noised input of the Generator and trick it as real data
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(100, 128)


# In[33]:


"""""self.AM = Sequential()
self.AM.add(self.generator())
self.AM.add(self.discriminator())
self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
metrics=['accuracy'])"""


# In[34]:


"""" def build_critic(self):

       model = Sequential()

       model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
       model.add(ZeroPadding2D(padding=((0,1),(0,1))))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
       model.add(BatchNormalization(momentum=0.8))
       model.add(LeakyReLU(alpha=0.2))
       model.add(Dropout(0.25))
       model.add(Flatten())
       model.add(Dense(1))

       model.summary()

       img = Input(shape=self.img_shape)
       validity = model(img)

       return Model(img, validity)
"""


# In[ ]:


"""" input = Input(img_shape)
   x =Conv2D(32, kernel_size=3, strides=2, padding="same")(input)
   x = LeakyReLU(alpha=0.2)(x)
   x = Dropout(0.25)(x)
   x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
   x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
   x = (LeakyReLU(alpha=0.2))(x)
   x = Dropout(0.25)(x)
   x = BatchNormalization(momentum=0.8)(x)
   x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
   x = LeakyReLU(alpha=0.2)(x)
   x = Dropout(0.25)(x)
   x = BatchNormalization(momentum=0.8)(x)
   x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
   x = LeakyReLU(alpha=0.2)(x)
   x = Dropout(0.25)(x)
   x = Flatten()(x)
   out = Dense(1, activation='sigmoid')(x)""""


# In[ ]:


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


# In[ ]:


""""real_size = (32,32,3)
z_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 25
alpha = 0.2
beta1 = 0.5


# In[ ]:


#batch normalization momentum

