"""Script for automating runs."""

import transfer_networks, naive_network, hyperparameters as hp

# Hyperparamters and file naming.
hp = hp.Hyperparameters()

hp.experiment_name = "testing"
hp.source_max_epochs = 30
hp.target_max_epochs = 100
hp.num_starting_units = 500
hp.upper_threshold = 0.9
hp.lower_threshold = 0.2
hp.source_lr = 0.001
hp.target_lr = 0.001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'

num_runs = 1
num_seeds = 2

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(num_runs):
        for i in range(num_seeds):
            print("Next seed = " + str(i))

            # Calls source network function
            seeded_units = transfer_networks.network(i, j, hp)
            # Calls naive network function
            naive_network.network(i, j, hp, seeded_units)
