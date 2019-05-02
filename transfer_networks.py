"""Cat detector source network which creates a dog detector target net."""
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
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.metrics import binary_accuracy
from keras.initializers import glorot_uniform

import library_extensions
import input_data
import utils


def network(seed, run, hp):
    cat_train_labels, cat_train_images, cat_val_labels, cat_val_images = input_data.get_training_and_val_data(
        hp.source_animal)
    cat_test_labels, cat_test_images = input_data.get_test_data(hp.source_animal)
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = input_data.get_training_and_val_data(
        hp.target_animal)
    dog_test_labels, dog_test_images = input_data.get_test_data(hp.target_animal)
    all_source_images = input_data.get_category_images(cat_train_labels, cat_train_images, 1)
    all_non_source_images = input_data.get_category_images(cat_train_labels, cat_train_images, 0)
    all_target_images = input_data.get_category_images(dog_train_labels, dog_train_images, 1)
    all_non_target_images = input_data.get_category_images(dog_train_labels, dog_train_images, 0)

    # Model
    model = Sequential()

    weight_init = glorot_uniform(seed)

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=cat_train_images.shape[1:], kernel_initializer=weight_init))
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
    model.add(Dense(hp.num_starting_units, kernel_initializer=weight_init, name="fc_layer"))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer=weight_init))
    model.add(Activation('sigmoid'))

    # Adam learning optimizer
    opt = keras.optimizers.adam(lr=hp.source_lr)

    # train the model using Adam
    model.compile(loss=hp.loss_function, optimizer=opt, metrics=[binary_accuracy])

    # Callbacks:
    # Stopping point value
    early_stopping = library_extensions.EarlyStoppingWithMax(target=1.1, monitor='val_binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.68)

    all_cat_predictions = library_extensions.PredictionHistory(model, cat_train_images, cat_train_labels,
                                                               cat_val_images, cat_val_labels, cat_test_images,
                                                               cat_test_labels)

    # Training source network
    model.fit(cat_train_images, cat_train_labels, batch_size=hp.batch_size, epochs=hp.source_max_epochs,
              validation_data=(dog_train_images, dog_train_labels), shuffle=True,
              callbacks=[all_cat_predictions])

    # Save stopped epoch variable
    if early_stopping.stopped_epoch == 0:
        cat_epoch_end = hp.source_max_epochs
    else:
        cat_epoch_end = early_stopping.stopped_epoch

    layer = model.get_layer("fc_layer")

    # Activation mean values
    activations = utils.get_activations(model, layer, all_target_images)
    if hp.pruning_method == 'activation':
        activations_mean = np.mean(activations, axis=0)
        discard_indices = \
            np.where((activations_mean <= hp.lower_threshold) | (activations_mean >= hp.upper_threshold))[0]

    # APoZ values
    apoz = library_extensions.get_apoz(model, layer, all_source_images)
    if hp.pruning_method == 'apoz':
        discard_indices = np.where((apoz <= hp.lower_threshold) | (apoz >= hp.upper_threshold))[0]

    # Creating the new target model.
    def my_delete_channels(model, layer, channels, *, node_indices=None):
        my_surgeon = library_extensions.MySurgeon(model)
        my_surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
        return my_surgeon.operate()

    # New model ready for training on dogs
    dogs_model = my_delete_channels(model, layer, discard_indices, node_indices=None)

    print(dogs_model.summary())

    # Adam learning optimizer
    dogs_opt = keras.optimizers.adam(lr=hp.target_lr)

    # train the model using Adam
    dogs_model.compile(loss=hp.loss_function, optimizer=dogs_opt, metrics=[binary_accuracy])

    # Callbacks:
    all_dog_predictions = library_extensions.PredictionHistory(dogs_model, dog_train_images, dog_train_labels,
                                                               dog_val_images, dog_val_labels, dog_test_images,
                                                               dog_test_labels)

    # Training target network
    dogs_model.fit(dog_train_images, dog_train_labels, epochs=hp.target_max_epochs,
                   batch_size=hp.batch_size, validation_data=(dog_val_images, dog_val_labels),
                   shuffle=True, callbacks=[all_dog_predictions])

    # Save number of neurons for use in naive network
    num_seeded_units = dogs_model.get_layer('fc_layer').get_config()['units']

    # Generate results
    run.seeded.update(seed, all_dog_predictions)
    run.update_single_data(seed, num_seeded_units, cat_epoch_end)
    run.update_apoz_data(seed, apoz)
    run.update_activation_data(seed, activations)

    return num_seeded_units
