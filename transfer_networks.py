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
import input_data as data
import results_data
import record_metadata

# Some hyperparameters for testing
batch_size = 200
cat_max_epochs = 30
dog_epochs = 10
num_starting_units = 300
upper_threshold = 0.9
lower_threshold = 0.2
cat_learning_rate = 0.0001
dog_learning_rate = 0.0001
conv_activation = 'relu'
loss_function = 'binary_crossentropy'

seed = 1
csv_directory_name = "testing"
experiment_number = "1"

# Saving the weights
save_dir = os.path.join(os.getcwd(), 'saved_models/' + csv_directory_name)
model_name = 'cat_model_' + experiment_number + str(seed) + '.h5'
dog_model_name = 'seeded_model_' + experiment_number + str(seed) + '.h5'

# Data gathering conditions
generate_graph = True
generate_cat_train_graph = False
network_name = "seeded"
cat_network_name = "cat"


def network(seed, units, csv_directory_name, experiment_number, dog_epochs, upper_threshold, lower_threshold,
            cat_learning_rate, dog_learning_rate, batch_size, conv_activation, loss_function):

    cat_train_labels, cat_train_images, cat_val_labels, cat_val_images = data.get_training_and_val_data('cat')
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = data.get_training_and_val_data('dog')
    dog_test_labels, dog_test_images = data.get_test_data('dog')

    # Model
    model = Sequential()

    weight_init = glorot_uniform(seed)

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=cat_train_images.shape[1:], kernel_initializer=weight_init))
    model.add(Activation(conv_activation))
    model.add(Conv2D(32, (3, 3), kernel_initializer=weight_init))
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=weight_init))
    model.add(Activation(conv_activation))
    model.add(Conv2D(64, (3, 3), kernel_initializer=weight_init))
    model.add(Activation(conv_activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units, kernel_initializer=weight_init, name="fc_layer"))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer=weight_init))
    model.add(Activation('sigmoid'))

    # Adam learning optimizer
    opt = keras.optimizers.adam(lr=cat_learning_rate)

    # train the model using Adam
    model.compile(loss=loss_function, optimizer=opt, metrics=[binary_accuracy])

    # Callbacks:
    # Built-in tensorflow data gathering at each epoch
    tensor_board_cats = TensorBoard(log_dir="logs/" + csv_directory_name + "/" + cat_network_name + "_" +
                                            experiment_number + str(seed))

    # Stopping point value
    early_stopping = library_extensions.EarlyStoppingWithMax(target=0.75, monitor='val_binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.68)

    # Training source network
    model.fit(cat_train_images, cat_train_labels, batch_size=batch_size, epochs=cat_max_epochs,
              validation_data=(dog_train_images, dog_train_labels), shuffle=True,
              callbacks=[tensor_board_cats])

    # Save stopped epoch variable
    if early_stopping.stopped_epoch == 0:
        cat_epoch_end = cat_max_epochs
    else:
        cat_epoch_end = early_stopping.stopped_epoch

    # Save model and weights:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Evaluate on Validation Data:
    predictions = model.predict_classes(cat_val_images)

    cat_stopping_MCC = str(matthews_corrcoef(cat_val_labels, predictions))

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
    if generate_cat_train_graph:
        cat_stopping_confusion = [cat_stopping_confusion]
        cat_stopping_MCC = [cat_stopping_MCC]

        np.array(cat_stopping_confusion)
        np.array(cat_stopping_MCC)

        np.savetxt("csv/CatConfusion/cat_results_" + str(seed) + ".csv",
                   np.column_stack((cat_stopping_confusion, cat_stopping_MCC)),
                   delimiter=",", fmt="%s")

    # APoZ values
    layer = model.get_layer("fc_layer")

    apoz = library_extensions.get_apoz(model, layer, cat_train_images)

    discard_indices = np.where((apoz <= lower_threshold) | (apoz >= upper_threshold))[0]

    # Creating the new target model.
    def my_delete_channels(model, layer, channels, *, node_indices=None):
        my_surgeon = library_extensions.MySurgeon(model)
        my_surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
        return my_surgeon.operate()

    # New model ready for training on dogs
    dogs_model = my_delete_channels(model, layer, discard_indices, node_indices=None)

    print(dogs_model.summary())

    # Adam learning optimizer
    dogs_opt = keras.optimizers.adam(lr=dog_learning_rate)

    # train the model using Adam
    dogs_model.compile(loss=loss_function, optimizer=dogs_opt, metrics=[binary_accuracy])

    # Callbacks:
    # Built-in tensorflow data gathering at each epoch
    dogs_tensor_board = TensorBoard(log_dir="logs/" + csv_directory_name + "/" + network_name + "_" +
                                            experiment_number + str(seed))

    dogs_early_stopping = library_extensions.EarlyStoppingWithMax(target=1.00, monitor='binary_accuracy', min_delta=0,
                                                                  patience=0, verbose=1, mode='auto', baseline=0.99)

    all_dog_predictions = library_extensions.PredictionHistory(generate_graph, model, dog_train_images,
                                                               dog_train_labels, dog_val_images, dog_val_labels,
                                                               dog_test_images, dog_test_labels)

    # Training target network
    dogs_model.fit(dog_train_images, dog_train_labels, epochs=dog_epochs,
                   batch_size=batch_size, validation_data=(dog_val_images, dog_val_labels),
                   shuffle=True, callbacks=[all_dog_predictions, dogs_tensor_board, dogs_early_stopping])

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

    # Generate MCC history csvs
    if generate_graph:
        generate_results = results_data.GenerateResults(model, network_name, all_dog_predictions, csv_directory_name,
                                                        experiment_number, seed)
        generate_results.generate_train_data(dog_train_images, dog_train_labels)
        generate_results.generate_val_data(dog_val_images, dog_val_labels)
        generate_results.generate_test_data(dog_test_images, dog_test_labels)

    # Save model and weights:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, dog_model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    record_metadata.record_metadata(csv_directory_name, experiment_number, seed, cat_max_epochs, num_seeded_units,
                                    lower_threshold, upper_threshold, cat_epoch_end, cat_learning_rate,
                                    dog_learning_rate, batch_size, conv_activation, loss_function)

    return num_seeded_units


if __name__ == "__main__":
    network(seed, num_starting_units, csv_directory_name, experiment_number, dog_epochs, upper_threshold, lower_threshold,
            cat_learning_rate, dog_learning_rate, batch_size, conv_activation, loss_function)
