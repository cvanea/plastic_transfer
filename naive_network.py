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
import hyperparameters as hp

# Some hyperparameters for testing
hp = hp.Hyperparameters()

hp.target_max_epochs = 3
hp.num_starting_units = 300
hp.upper_threshold = 0.9
hp.lower_threshold = 0.2
hp.target_lr = 0.0001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'

hp.experiment_name = "testing"
run_num = 1
seed = 2
network_name = "naive"

# Saving the weights.
save_dir = os.path.join(os.getcwd(), 'results/' + hp.experiment_name + "/run_" + str(run_num) + "/" + network_name)
model_name = network_name + '_model_seed_' + str(seed) + '.h5'

# Data gathering conditions
generate_mcc_results = True
generate_accuracy_results = True
save_dog_model = False


def network(seed, run_num, hp, num_seeded_units):
    dog_train_labels, dog_train_images, dog_val_labels, dog_val_images = data.get_training_and_val_data('dog')
    dog_test_labels, dog_test_images = data.get_test_data("dog")

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
    # Built-in tensorflow data gathering at each epoch
    tensor_board = TensorBoard(
        log_dir="results/" + hp.experiment_name + "/run_" + str(run_num) + "/" + network_name + "/logs/seed_" + str(
            seed))

    early_stopping = library_extensions.EarlyStoppingWithMax(target=1.00, monitor='binary_accuracy', min_delta=0,
                                                             patience=0, verbose=1, mode='auto', baseline=0.99)

    all_predictions = library_extensions.PredictionHistory(generate_mcc_results, generate_accuracy_results, model,
                                                           dog_train_images, dog_train_labels, dog_val_images,
                                                           dog_val_labels, dog_test_images, dog_test_labels)

    # Training naive network
    history = model.fit(dog_train_images, dog_train_labels, batch_size=hp.batch_size, epochs=hp.target_max_epochs,
                        validation_data=(dog_val_images, dog_val_labels), shuffle=True,
                        callbacks=[all_predictions, tensor_board])

    # Save model and weights:
    if save_dog_model:
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

    # Generate results history
    if generate_mcc_results or generate_accuracy_results:
        generate_results = results_data.GenerateResults(model, network_name, all_predictions, history,
                                                        hp.experiment_name, run_num, seed)
        if generate_mcc_results:
            generate_results.generate_train_data('mcc')
            generate_results.generate_val_data('mcc')
            generate_results.generate_test_data('mcc')
        if generate_accuracy_results:
            generate_results.generate_train_data('accuracy')
            generate_results.generate_val_data('accuracy')
            generate_results.generate_test_data('accuracy')


if __name__ == "__main__":
    network(seed, run_num, hp)
