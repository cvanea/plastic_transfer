"""Naive dog detector network."""

from numpy.random import seed

seed(1)
import os

os.environ['PYTHONHASHSEED'] = '0'
import random

random.seed(1)
import tensorflow as tf
from keras import backend as K

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.metrics import binary_accuracy
from keras.initializers import glorot_uniform

import input_data as data
import library_extensions

run_num = 1
seed = 0
network_name = "naive"


def network(seed, run, hp, num_seeded_units):
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = data.get_training_and_val_data(
        hp.target_animal)
    dog_test_labels, dog_test_images = data.get_test_data(hp.target_animal)

    # Model
    model = Sequential()

    weight_init = glorot_uniform(seed)

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=dog_train_images.shape[1:], kernel_initializer=weight_init))
    model.add(Activation(hp.conv_activation))
    model.add(Conv2D(32, (3, 3), kernel_initializer=weight_init))
    model.add(Activation(hp.conv_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=weight_init))
    model.add(Activation(hp.conv_activation))
    model.add(Conv2D(64, (3, 3), kernel_initializer=weight_init))
    model.add(Activation(hp.conv_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(num_seeded_units, kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer=weight_init))
    model.add(Activation('sigmoid'))

    # Adam learning optimizer
    opt = keras.optimizers.adam(lr=hp.target_lr)

    # train the model using Adam
    model.compile(loss=hp.loss_function, optimizer=opt, metrics=[binary_accuracy])

    # Callbacks:
    all_predictions = library_extensions.PredictionHistory(model, dog_train_images, dog_train_labels, dog_val_images,
                                                           dog_val_labels, dog_test_images, dog_test_labels)

    # Training naive network
    model.fit(dog_train_images, dog_train_labels, batch_size=hp.batch_size, epochs=hp.target_max_epochs,
              validation_data=(dog_val_images, dog_val_labels), shuffle=True,
              callbacks=[all_predictions])

    # Generate results history
    run.naive.update(seed, all_predictions)
