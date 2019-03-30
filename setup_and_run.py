"""Script for automating runs."""

import transfer_networks, naive_network

# Hyperparamters and file naming.
num_results = 1
csv_directory_name = "testing"
experiment_number = "1"
batch_size = 200
target_epochs = 30
num_starting_units = 500
upper_threshold = 0.9
lower_threshold = 0.2
cat_learning_rate = 0.001
dog_learning_rate = 0.001
conv_activation = 'relu'
loss_function = 'binary_crossentropy'


if __name__ == "__main__":

    # Generate num_results number of weight initialisation runs.
    for i in range(num_results):
        print("Next seed = " + str(i))

        # Calls source network function
        seeded_units = transfer_networks.network(i, num_starting_units, csv_directory_name, experiment_number,
                                                 target_epochs, upper_threshold, lower_threshold, cat_learning_rate,
                                                 dog_learning_rate, batch_size, conv_activation, loss_function)
        # Calls naive network function
        naive_network.network(i, seeded_units, csv_directory_name, experiment_number, target_epochs,
                              dog_learning_rate, batch_size, conv_activation, loss_function)
