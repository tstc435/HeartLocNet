'''Create a simple deep CNN mode used on the X-CHEST small images dataset.
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import sys
sys.path.append('dataprovider/')
from chestdataset import ChestDataSet

num_classes = 2
num_predictions = 2

class ChestNetModeS:
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def create_mode(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(64,64,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        opt = keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, decay=1e-6)

        # Let's train the model using Adam
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model.summary()
        for l in model.layers:
            print (l.output_shape)
        return model

if __name__ == '__main__':
    chestNetMode = ChestNetModeS()
    chestNetMode.create_mode()
