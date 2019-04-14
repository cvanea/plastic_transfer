"""Script for automating runs locally."""

import transfer_networks, naive_network, hyperparameters as hp
from run import Run
from gen_graphs import gen_graphs

hp = hp.Hyperparameters()

hp.source_max_epochs = 1
hp.target_max_epochs = 1
hp.num_starting_units = 500
hp.upper_threshold = 0.9
hp.lower_threshold = 0.2
hp.source_lr = 0.001
hp.target_lr = 0.001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'

num_runs = 1
num_seeds = 3

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(num_runs):
        run = Run('testing2', j, hp)

        for i in range(num_seeds):
            print("Next seed = " + str(i))

            # Calls source network function
            seeded_units = transfer_networks.network(i, run, hp)
            run.save(naive=False)

            # Calls naive network function
            naive_network.network(i, run, hp, seeded_units)
            run.save(seeded=False)

        gen_graphs(run)
