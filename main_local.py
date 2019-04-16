"""Script for automating runs locally."""

import transfer_networks, naive_network, hyperparameters as hp
from run import Run
from gen_graphs import gen_graphs
from utils import save_results_to_bucket

hp = hp.Hyperparameters()
hp.source_max_epochs = 2
hp.target_max_epochs = 2
hp.num_starting_units = 500
hp.upper_threshold = 0.9
hp.lower_threshold = 0.2
hp.source_lr = 0.001
hp.target_lr = 0.001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'

run_0 = Run('testing3', 0, hp)
# run_1 = Run('testing3', 1, hp)
# run_2 = Run('testing3', 2, hp)
# run_1.hyperparameters.num_starting_units = 700
# run_2.hyperparameters.num_starting_units = 1000

runs = [run_0]

# num_runs = 1
num_seeds = 2

seeded = True
naive = True

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(len(runs)):
        if seeded:
            run = runs[j]
        else:
            run = Run.restore('testing3', j, naive=False)

        for i in range(num_seeds):
            print("Next seed = " + str(i))

            # Calls source network function
            if seeded:
                transfer_networks.network(i, run, hp)
                run.save(naive=False)

            # Calls naive network function
            if naive:
                seeded_units = run.single_data.loc['num_seeded_units', str(i)]
                naive_network.network(i, run, hp, seeded_units)
                run.save(seeded=False)

        gen_graphs(run, seeded=seeded, naive=naive)

    # save_results_to_bucket()
