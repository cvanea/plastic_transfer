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
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import input_data as data
import results_data
import library_extensions
import record_metadata

# Some hyperparameters
batch_size = 200
outputs = 1
epochs = 10
dog_learning_rate = 0.0001
seed = 1
units = 300
csv_directory_name = "testing"
experiment_number = "1."

# Saving the weights.
save_dir = os.path.join(os.getcwd(), 'saved_models/' + csv_directory_name)
model_name = 'naive_model_' + experiment_number + str(seed) + '.h5'

# Data gathering conditions
generate_graphs = True
network_name = "naive"


def network(seed, units, csv_directory_name, experiment_number, epochs, dog_learning_rate):
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = data.get_training_and_val_data('dog')
    dog_test_labels, dog_test_images = data.get_test_data("dog")

    # Model
    model = Sequential()

    weight_init = glorot_uniform(seed)

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=dog_train_images.shape[1:], kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units, kernel_initializer=weight_init))
    model.add(Activation('relu'))
    model.add(Dense(outputs, kernel_initializer=weight_init))
    model.add(Activation('sigmoid'))

    # Adam learning optimizer
    opt = keras.optimizers.adam(lr=dog_learning_rate)

    # train the model using Adam
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[binary_accuracy])

    # Callbacks:
    # Built-in tensorflow data gathering at each epoch
    tensor_board = TensorBoard(log_dir="logs/naive_dogs/" + csv_directory_name + "_" + experiment_number +
                                       str(seed))

    early_stopping = library_extensions.EarlyStoppingWithMax(target=1.00, monitor='binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.99)

    all_predictions = library_extensions.PredictionHistory(generate_graphs, model, dog_train_images, dog_train_labels,
                                                           dog_val_images, dog_val_labels, dog_test_images,
                                                           dog_test_labels)

    # Training naive network
    model.fit(dog_train_images, dog_train_labels, batch_size=batch_size, epochs=epochs,
              validation_data=(dog_val_images, dog_val_labels), shuffle=True,
              callbacks=[all_predictions, early_stopping, tensor_board])

    # Save model and weights:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Evaluate:
    scores = model.evaluate(dog_test_images, dog_test_labels, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    predictions = model.predict_classes(dog_test_images)

    print("Naive Dogs Evaluation:")
    print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(dog_test_labels, predictions)))

    print("True Negative, False Positive, False Negative, True Positive:")
    print(confusion_matrix(dog_test_labels, predictions).ravel())

    # Generate MCC history csvs
    if generate_graphs:
        generate_results = results_data.GenerateResults(model, network_name, all_predictions, csv_directory_name,
                                                        experiment_number, seed)
        generate_results.generate_train_data(dog_train_images, dog_train_labels)
        generate_results.generate_val_data(dog_val_images, dog_val_labels)
        generate_results.generate_test_data(dog_test_images, dog_test_labels)

    record_metadata.record_metadata()


if __name__ == "__main__":
    network(seed, units, csv_directory_name, experiment_number, epochs, dog_learning_rate)
