"""Script for automating runs locally."""

import transfer_networks, naive_network, hyperparameters as hyperparam
from run import Run
from gen_graphs import gen_graphs
from utils import save_results_to_bucket

hp = hyperparam.Hyperparameters()
hp.source_max_epochs = 15
hp.target_max_epochs = 1
hp.num_starting_units = 100
hp.upper_threshold = 1.0
hp.lower_threshold = 0.0
hp.source_lr = 0.001
hp.target_lr = 0.001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'
hp.pruning_method = 'activation'
hp.source_animal = "cat"
hp.target_animal = "dog"

run_1 = Run('exp_8', 1, hp)

hp2 = hyperparam.Hyperparameters()
hp2.source_max_epochs = 30
hp2.target_max_epochs = 1
hp2.num_starting_units = 100
hp2.upper_threshold = 1.1
hp2.lower_threshold = 0.0
hp2.source_lr = 0.001
hp2.target_lr = 0.001
hp2.batch_size = 200
hp2.conv_activation = 'relu'
hp2.loss_function = 'binary_crossentropy'
hp2.pruning_method = 'activation'
hp2.source_animal = "cat"
hp2.target_animal = "dog"

run_2 = Run('exp_8', 2, hp2)

hp3 = hyperparam.Hyperparameters()
hp3.source_max_epochs = 45
hp3.target_max_epochs = 1
hp3.num_starting_units = 100
hp3.upper_threshold = 1.1
hp3.lower_threshold = 0.0
hp3.source_lr = 0.001
hp3.target_lr = 0.001
hp3.batch_size = 200
hp3.conv_activation = 'relu'
hp3.loss_function = 'binary_crossentropy'
hp3.pruning_method = 'activation'
hp3.source_animal = "cat"
hp3.target_animal = "dog"

run_3 = Run('exp_8', 3, hp3)

hp4 = hyperparam.Hyperparameters()
hp4.source_max_epochs = 60
hp4.target_max_epochs = 1
hp4.num_starting_units = 100
hp4.upper_threshold = 1.1
hp4.lower_threshold = 0.0
hp4.source_lr = 0.001
hp4.target_lr = 0.001
hp4.batch_size = 200
hp4.conv_activation = 'relu'
hp4.loss_function = 'binary_crossentropy'
hp4.pruning_method = 'activation'
hp4.source_animal = "cat"
hp4.target_animal = "dog"

run_4 = Run('exp_8', 4, hp4)

hp5 = hyperparam.Hyperparameters()
hp5.source_max_epochs = 75
hp5.target_max_epochs = 1
hp5.num_starting_units = 100
hp5.upper_threshold = 1.1
hp5.lower_threshold = 0.0
hp5.source_lr = 0.001
hp5.target_lr = 0.001
hp5.batch_size = 200
hp5.conv_activation = 'relu'
hp5.loss_function = 'binary_crossentropy'
hp5.pruning_method = 'activation'
hp5.source_animal = "cat"
hp5.target_animal = "dog"

run_5 = Run('exp_8', 5, hp5)

hp6 = hyperparam.Hyperparameters()
hp6.source_max_epochs = 90
hp6.target_max_epochs = 1
hp6.num_starting_units = 100
hp6.upper_threshold = 1.1
hp6.lower_threshold = 0.0
hp6.source_lr = 0.001
hp6.target_lr = 0.001
hp6.batch_size = 200
hp6.conv_activation = 'relu'
hp6.loss_function = 'binary_crossentropy'
hp6.pruning_method = 'activation'
hp6.source_animal = "cat"
hp6.target_animal = "dog"

run_6 = Run('exp_8', 6, hp6)

runs = [run_1, run_2, run_3, run_4, run_5, run_6]

# num_runs = 1
num_seeds = 3

seeded = True
naive = False

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(len(runs)):
        if seeded:
            run = runs[j]
        elif naive:
            run = Run.restore('test', j, num_seeds, naive=False)
        else:
            run = Run.restore('test', 1, num_seeds)
            gen_graphs(run)

        if seeded or naive:
            for i in range(num_seeds):
                print("Current seed = " + str(i))

                # Calls source network function
                if seeded:
                    transfer_networks.network(i, run, run.hyperparameters)
                    run.save(naive=False)

                # Calls naive network function
                if naive:
                    seeded_units = run.single_data.loc['num_seeded_units', str(i)]
                    naive_network.network(i, run, run.hyperparameters, seeded_units)
                    run.save(seeded=False)

            gen_graphs(run, seeded=seeded, naive=naive)

    save_results_to_bucket()