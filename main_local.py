"""Script for automating runs locally."""

import transfer_networks, naive_network, hyperparameters as hyperparam
from run import Run
from gen_graphs import gen_graphs
from utils import save_results_to_bucket

hp = hyperparam.Hyperparameters()
hp.source_max_epochs = 15
hp.target_max_epochs = 400
hp.num_starting_units = 100
hp.upper_threshold = 10.0
hp.lower_threshold = 0.1
hp.source_lr = 0.001
hp.target_lr = 0.0001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'
hp.pruning_method = 'thresh_maru'
hp.source_animal = "cat"
hp.target_animal = "dog"
hp.pruning_dataset = "p_target"
hp.save_opp = True

run_1 = Run('exp_10', 4, hp, save_opp=hp.save_opp)

hp2 = hyperparam.Hyperparameters()
hp2.source_max_epochs = 30
hp2.target_max_epochs = 1
hp2.num_starting_units = 100
hp2.upper_threshold = 10.0
hp2.lower_threshold = -0.1
hp2.source_lr = 0.001
hp2.target_lr = 0.001
hp2.batch_size = 200
hp2.conv_activation = 'relu'
hp2.loss_function = 'binary_crossentropy'
hp2.pruning_method = 'thresh_maru'
hp2.source_animal = "cat"
hp2.target_animal = "dog"
hp2.pruning_dataset = "p_target"
hp2.save_opp = True

run_2 = Run('exp_10', 2, hp2, save_opp=hp2.save_opp)

runs = [run_1]

num_seeds = 3

target = True
naive = True

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(len(runs)):
        if target:
            run = runs[j]
        elif naive:
            run = Run.restore('test', 2, num_seeds, naive=False, save_opp=False)
        else:
            run = Run.restore('test', 2, num_seeds, save_opp=False)
            gen_graphs(run, save_opp=False)

        if target or naive:
            for i in range(num_seeds):
                print("Current seed = " + str(i))

                # Calls source network function
                if target:
                    transfer_networks.network(i, run, run.hyperparameters)
                    run.save(naive=False)

                # Calls naive network function
                if naive:
                    seeded_units = run.single_data.loc['num_seeded_units', str(i)]
                    naive_network.network(i, run, run.hyperparameters, seeded_units)
                    run.save(target=False)

            gen_graphs(run, target=target, naive=naive, save_opp=run.hyperparameters.save_opp)

    save_results_to_bucket()