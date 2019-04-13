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
import input_data
import results_data
import hyperparameters as hp

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

hp.experiment_name = "testing"
run_num = 1
seed = 4
network_name = "seeded"
cat_network_name = "cat"

# Saving the weights
save_dir = os.path.join(os.getcwd(), 'results/' + hp.experiment_name + "/run_" + str(run_num) + "/" + network_name)
dog_model_name = network_name + '_model_seed_' + str(seed) + '.h5'
model_name = network_name + '_model_seed_' + str(seed) + '.h5'

# Data gathering conditions
generate_mcc_results = False
generate_accuracy_results = False
generate_cat_train_results = False
save_cat_model = False
save_dog_model = False


def network(seed, run_num, hp):
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
        log_dir="results/" + hp.experiment_name + "/run_" + str(run_num) + "/" + cat_network_name + "/logs/seed_" + str(
            seed))

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
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
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
        log_dir="results/" + hp.experiment_name + "/run_" + str(run_num) + "/" + network_name + "/logs/seed_" + str(
            seed))

    dogs_early_stopping = library_extensions.EarlyStoppingWithMax(target=1.00, monitor='binary_accuracy', min_delta=0,
                                                                  patience=0, verbose=1, mode='auto', baseline=0.99)

    all_dog_predictions = library_extensions.PredictionHistory(generate_mcc_results, generate_accuracy_results,
                                                               model, dog_train_images, dog_train_labels,
                                                               dog_val_images, dog_val_labels, dog_test_images,
                                                               dog_test_labels)

    # Training target network
    dog_history = dogs_model.fit(dog_train_images, dog_train_labels, epochs=hp.target_max_epochs,
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
        generate_results = results_data.GenerateResults(network_name, all_dog_predictions, dog_history,
                                                        hp.experiment_name, run_num, seed)
        generate_results.generate_seeded_units_stopped_epoch_data(num_seeded_units, cat_epoch_end)
        if generate_mcc_results:
            generate_results.generate_train_data('mcc')
            generate_results.generate_val_data('mcc')
            generate_results.generate_test_data('mcc')
        if generate_accuracy_results:
            generate_results.generate_train_data('accuracy')
            generate_results.generate_val_data('accuracy')
            generate_results.generate_test_data('accuracy')

    # Save model and weights:
    if save_dog_model:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, dog_model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    # Save parameters if they aren't already saved
    if not os.path.exists("results/" + hp.experiment_name + "/params.csv"):
        hp.to_csv(run_num)

    return num_seeded_units


if __name__ == "__main__":
    network(seed, run_num, hp)
