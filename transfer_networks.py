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
from sklearn.preprocessing import minmax_scale

import library_extensions
import input_data
import utils


def network(seed, run, hp):
    source_train_labels, source_train_images, source_val_labels, source_val_images = input_data.get_training_and_val_data(
        hp.source_animal)
    source_test_labels, source_test_images = input_data.get_test_data(hp.source_animal)
    target_train_labels, target_train_images, target_val_labels, target_val_images = input_data.get_training_and_val_data(
        hp.target_animal)
    target_test_labels, target_test_images = input_data.get_test_data(hp.target_animal)
    if hp.pruning_dataset == 'p_source':
        pruning_dataset = input_data.get_category_images(source_train_labels, source_train_images, 1)
    elif hp.pruning_dataset == 'n_source':
        pruning_dataset = input_data.get_category_images(source_train_labels, source_train_images, 0)
    elif hp.pruning_dataset == 'p_target':
        pruning_dataset = input_data.get_category_images(target_train_labels, target_train_images, 1)
    else:
        pruning_dataset = input_data.get_category_images(target_train_labels, target_train_images, 0)

    # Model
    source_model = Sequential()

    weight_init = glorot_uniform(seed)

    source_model.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=source_train_images.shape[1:], kernel_initializer=weight_init))
    source_model.add(Activation(hp.conv_activation))
    source_model.add(Conv2D(32, (3, 3), kernel_initializer=weight_init))
    source_model.add(Activation(hp.conv_activation))
    source_model.add(MaxPooling2D(pool_size=(2, 2)))
    source_model.add(Dropout(0.25))

    source_model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=weight_init))
    source_model.add(Activation(hp.conv_activation))
    source_model.add(Conv2D(64, (3, 3), kernel_initializer=weight_init))
    source_model.add(Activation(hp.conv_activation))
    source_model.add(MaxPooling2D(pool_size=(2, 2)))
    source_model.add(Dropout(0.25))

    source_model.add(Flatten())
    source_model.add(Dense(hp.num_starting_units, kernel_initializer=weight_init, name="fc_layer"))
    source_model.add(Activation('relu'))
    source_model.add(Dense(1, kernel_initializer=weight_init))
    source_model.add(Activation('sigmoid'))

    # Adam learning optimizer
    opt = keras.optimizers.adam(lr=hp.source_lr)

    # train the source_model using Adam
    source_model.compile(loss=hp.loss_function, optimizer=opt, metrics=[binary_accuracy])

    # Callbacks:
    # Stopping point value
    early_stopping = library_extensions.EarlyStoppingWithMax(target=1.1, monitor='val_binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.68)

    all_source_predictions = library_extensions.PredictionHistory(source_model, source_train_images,
                                                                  source_train_labels,
                                                                  source_val_images, source_val_labels,
                                                                  source_test_images,
                                                                  source_test_labels, save_opp=hp.save_opp,
                                                                  opp_train_images=target_train_images,
                                                                  opp_train_labels=target_train_labels,
                                                                  opp_val_images=target_val_images,
                                                                  opp_val_labels=target_val_labels,
                                                                  opp_test_images=target_test_images,
                                                                  opp_test_labels=target_test_labels)

    # Training source network
    source_model.fit(source_train_images, source_train_labels, batch_size=hp.batch_size, epochs=hp.source_max_epochs,
                     validation_data=(target_train_images, target_train_labels), shuffle=True,
                     callbacks=[all_source_predictions])

    # Save stopped epoch variable
    if early_stopping.stopped_epoch == 0:
        cat_epoch_end = hp.source_max_epochs
    else:
        cat_epoch_end = early_stopping.stopped_epoch

    layer = source_model.get_layer("fc_layer")

    # Finding neuron indicies to prune
    apoz = library_extensions.get_apoz(source_model, layer, pruning_dataset)
    activations = utils.get_activations(source_model, layer, pruning_dataset)

    # Mean activation method
    if hp.pruning_method == 'activation':
        activations_mean = np.mean(activations, axis=0)
        discard_indices = \
            np.where((activations_mean <= hp.lower_threshold) | (activations_mean >= hp.upper_threshold))[0]

    # Normalised threshold method
    elif hp.pruning_method == 'thresh_maru':
        mean_data = np.mean(activations, axis=0)
        std_data = np.std(activations, axis=0)
        maru_data = np.divide(mean_data, std_data, out=np.zeros_like(mean_data), where=std_data != 0)
        normalised_maru = minmax_scale(maru_data, feature_range=(0, 1))
        discard_indices = \
            np.where((normalised_maru <= hp.lower_threshold) | (normalised_maru >= hp.upper_threshold))[0]

    # APoZ method
    else:
        discard_indices = np.where((apoz <= hp.lower_threshold) | (apoz >= hp.upper_threshold))[0]

    # Creating the new target source_model.
    def my_delete_channels(model, layer, channels, *, node_indices=None):
        my_surgeon = library_extensions.MySurgeon(model)
        my_surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
        return my_surgeon.operate()

    # New source_model ready for training on dogs
    target_model = my_delete_channels(source_model, layer, discard_indices, node_indices=None)

    print(target_model.summary())

    # Adam learning optimizer
    target_opt = keras.optimizers.adam(lr=hp.target_lr)

    # train the source_model using Adam
    target_model.compile(loss=hp.loss_function, optimizer=target_opt, metrics=[binary_accuracy])

    # Callbacks:
    all_target_predictions = library_extensions.PredictionHistory(target_model, target_train_images,
                                                                  target_train_labels, target_val_images,
                                                                  target_val_labels, target_test_images,
                                                                  target_test_labels, save_opp=hp.save_opp,
                                                                  opp_train_images=source_train_images,
                                                                  opp_train_labels=source_train_labels,
                                                                  opp_val_images=source_val_images,
                                                                  opp_val_labels=source_val_labels,
                                                                  opp_test_images=source_test_images,
                                                                  opp_test_labels=source_test_labels)

    # Training target network
    target_model.fit(target_train_images, target_train_labels, epochs=hp.target_max_epochs,
                     batch_size=hp.batch_size, validation_data=(target_val_images, target_val_labels),
                     shuffle=True, callbacks=[all_target_predictions])

    # Save number of neurons for use in naive network
    num_seeded_units = target_model.get_layer('fc_layer').get_config()['units']

    # Generate results
    run.save_opp = hp.save_opp
    run.source.update(seed, all_source_predictions)
    run.target.update(seed, all_target_predictions)
    run.update_single_data(seed, num_seeded_units, cat_epoch_end)
    run.update_apoz_data(seed, apoz)
    run.update_activation_data(seed, activations)

    return num_seeded_units
