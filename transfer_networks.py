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
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import library_extensions
from utils import create_path
import input_data
import hyperparameters as hp
from run import Run

# Some hyperparameters for testing
hp = hp.Hyperparameters()

hp.source_max_epochs = 2
hp.target_max_epochs = 2
hp.num_starting_units = 300
hp.upper_threshold = 0.9
hp.lower_threshold = 0.2
hp.source_lr = 0.0001
hp.target_lr = 0.0001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'

run_num = 1
seed = 0
network_name = "seeded"
cat_network_name = "cat"

run = Run("testing", 1, hp)

# Saving the weights
save_dir = run.path + "/" + network_name
dog_model_name = network_name + '_model_seed_' + str(seed) + '.h5'
cat_model_name = cat_network_name + '_model_seed_' + str(seed) + '.h5'

# Data gathering conditions
generate_mcc_results = True
generate_accuracy_results = True
generate_cat_train_results = False
save_cat_model = False
save_dog_model = False


def network(seed, run, hp):
    cat_train_labels, cat_train_images, cat_val_labels, cat_val_images = input_data.get_training_and_val_data('cat')
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = input_data.get_training_and_val_data('dog')
    dog_test_labels, dog_test_images = input_data.get_test_data('dog')

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
    # Built-in tensorflow data gathering at each epoch
    tensor_board_cats = TensorBoard(
        log_dir=run.path + "/" + cat_network_name + "/logs/seed_" + str(seed))

    # Stopping point value
    early_stopping = library_extensions.EarlyStoppingWithMax(target=0.74, monitor='val_binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.68)

    # Training source network
    model.fit(cat_train_images, cat_train_labels, batch_size=hp.batch_size, epochs=hp.source_max_epochs,
              validation_data=(dog_train_images, dog_train_labels), shuffle=True,
              callbacks=[tensor_board_cats, early_stopping])

    # Save stopped epoch variable
    if early_stopping.stopped_epoch == 0:
        cat_epoch_end = hp.source_max_epochs
    else:
        cat_epoch_end = early_stopping.stopped_epoch

    # Save model and weights:
    if save_cat_model:
        model_path = create_path(save_dir, cat_model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    # Evaluate on Validation Data:
    predictions = model.predict_classes(cat_val_images)

    cat_stopping_mcc = str(matthews_corrcoef(cat_val_labels, predictions))

    print("Cat Evaluation:")
    print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(cat_val_labels, predictions)))

    cat_stopping_confusion = confusion_matrix(cat_val_labels, predictions).ravel()
    print("True Negative, False Positive, False Negative, True Positive:")
    print(confusion_matrix(cat_val_labels, predictions).ravel())

    print("Dog Evaluation:")
    print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(dog_val_labels, predictions)))

    print("True Negative, False Positive, False Negative, True Positive:")
    print(confusion_matrix(dog_val_labels, predictions).ravel())

    # Old generates data at stopping point:
    if generate_cat_train_results:
        cat_stopping_confusion = [cat_stopping_confusion]
        cat_stopping_mcc = [cat_stopping_mcc]

        np.array(cat_stopping_confusion)
        np.array(cat_stopping_mcc)

        np.savetxt("csv/CatConfusion/cat_results_" + str(seed) + ".csv",
                   np.column_stack((cat_stopping_confusion, cat_stopping_mcc)),
                   delimiter=",", fmt="%s")

    # APoZ values
    layer = model.get_layer("fc_layer")

    apoz = library_extensions.get_apoz(model, layer, cat_train_images)

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
    # Built-in tensorflow data gathering at each epoch
    dogs_tensor_board = TensorBoard(
        log_dir=run.path + "/" + network_name + "/logs/seed_" + str(seed))

    dogs_early_stopping = library_extensions.EarlyStoppingWithMax(target=1.00, monitor='binary_accuracy', min_delta=0,
                                                                  patience=0, verbose=1, mode='auto', baseline=0.99)

    all_dog_predictions = library_extensions.PredictionHistory(generate_mcc_results, generate_accuracy_results,
                                                               model, dog_train_images, dog_train_labels,
                                                               dog_val_images, dog_val_labels, dog_test_images,
                                                               dog_test_labels)

    # Training target network
    dogs_model.fit(dog_train_images, dog_train_labels, epochs=hp.target_max_epochs,
                   batch_size=hp.batch_size, validation_data=(dog_val_images, dog_val_labels),
                   shuffle=True, callbacks=[all_dog_predictions, dogs_tensor_board])

    # Predictions on training data
    dog_train_predictions = dogs_model.predict_classes(dog_train_images)

    print("Training Data - Dog Evaluation:")
    print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(dog_train_labels, dog_train_predictions)))

    print("True Negative, False Positive, False Negative, True Positive:")
    print(confusion_matrix(dog_train_labels, dog_train_predictions).ravel())

    # Scores on test data
    dogs_scores = dogs_model.evaluate(dog_test_images, dog_test_labels, verbose=1)
    print('Final test dog loss:', dogs_scores[0])
    print('Final test dog accuracy:', dogs_scores[1])

    # Predictions on balanced test data
    dog_predictions = dogs_model.predict_classes(dog_test_images)

    # Evaluate:
    print("Test Data - Dog Evaluation:")
    print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(dog_test_labels, dog_predictions)))

    print("True Negative, False Positive, False Negative, True Positive:")
    print(confusion_matrix(dog_test_labels, dog_predictions).ravel())

    # Save number of neurons for use in naive network
    num_seeded_units = dogs_model.get_layer('fc_layer').get_config()['units']

    # Generate results
    if generate_mcc_results or generate_accuracy_results:
        run.seeded.update(seed, all_dog_predictions)
        run.update_single_data(seed, num_seeded_units, cat_epoch_end)

    # Save model and weights:
    if save_dog_model:
        model_path = create_path(save_dir, dog_model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    return num_seeded_units


if __name__ == "__main__":
    network(seed, run, hp)
