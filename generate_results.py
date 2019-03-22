"""Script for data gathering."""

import transfer_networks, naive_network

# Hyperparamters and file naming.
num_results = 1
csv_directory_name = "testing"
experiment_number = "2."
epochs = 30
units = 500
upper_threshold = 0.9
lower_threshold = 0.2
cat_learning_rate = 0.001
dog_learning_rate = 0.001


if __name__ == "__main__":

    # Generate num_results number of weight initialisation runs.
    for i in range(num_results):
        print("Next seed = " + str(i))

        # Calls source network function
        seeded_units = transfer_networks.network(i, units, csv_directory_name, experiment_number, epochs,
                                                 upper_threshold, lower_threshold, cat_learning_rate, dog_learning_rate)
        # Calls naive network function
        naive_network.network(i, seeded_units, csv_directory_name, experiment_number, epochs, dog_learning_rate)
