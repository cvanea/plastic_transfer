"""Script for automating runs locally."""

import transfer_networks, naive_network, hyperparameters as hyperparam
from run import Run
from gen_graphs import gen_graphs
from utils import save_results_to_bucket

hp = hyperparam.Hyperparameters()
hp.source_max_epochs = 15
hp.target_max_epochs = 400
hp.num_starting_units = 100
hp.upper_threshold = 20.0
hp.lower_threshold = -0.1
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

run_1 = Run('exp_11', 1, hp, save_opp=hp.save_opp)

hp2 = hyperparam.Hyperparameters()
hp2.source_max_epochs = 15
hp2.target_max_epochs = 400
hp2.num_starting_units = 100
hp2.upper_threshold = 20.0
hp2.lower_threshold = -0.1
hp2.source_lr = 0.001
hp2.target_lr = 0.0001
hp2.batch_size = 200
hp2.conv_activation = 'relu'
hp2.loss_function = 'binary_crossentropy'
hp2.pruning_method = 'thresh_maru'
hp2.source_animal = "cat"
hp2.target_animal = "horse"
hp2.pruning_dataset = "p_target"
hp2.save_opp = True

run_2 = Run('exp_11', 2, hp2, save_opp=hp2.save_opp)

hp3 = hyperparam.Hyperparameters()
hp3.source_max_epochs = 15
hp3.target_max_epochs = 400
hp3.num_starting_units = 100
hp3.upper_threshold = 20.0
hp3.lower_threshold = -0.1
hp3.source_lr = 0.001
hp3.target_lr = 0.0001
hp3.batch_size = 200
hp3.conv_activation = 'relu'
hp3.loss_function = 'binary_crossentropy'
hp3.pruning_method = 'thresh_maru'
hp3.source_animal = "cat"
hp3.target_animal = "ship"
hp3.pruning_dataset = "p_target"
hp3.save_opp = True

run_3 = Run('exp_11', 3, hp3, save_opp=hp3.save_opp)

hp4 = hyperparam.Hyperparameters()
hp4.source_max_epochs = 15
hp4.target_max_epochs = 400
hp4.num_starting_units = 100
hp4.upper_threshold = 20.0
hp4.lower_threshold = -0.1
hp4.source_lr = 0.001
hp4.target_lr = 0.0001
hp4.batch_size = 200
hp4.conv_activation = 'relu'
hp4.loss_function = 'binary_crossentropy'
hp4.pruning_method = 'thresh_maru'
hp4.source_animal = "horse"
hp4.target_animal = "deer"
hp4.pruning_dataset = "p_target"
hp4.save_opp = True

run_4 = Run('exp_11', 4, hp4, save_opp=hp4.save_opp)

hp5 = hyperparam.Hyperparameters()
hp5.source_max_epochs = 15
hp5.target_max_epochs = 400
hp5.num_starting_units = 100
hp5.upper_threshold = 20.0
hp5.lower_threshold = -0.1
hp5.source_lr = 0.001
hp5.target_lr = 0.0001
hp5.batch_size = 200
hp5.conv_activation = 'relu'
hp5.loss_function = 'binary_crossentropy'
hp5.pruning_method = 'thresh_maru'
hp5.source_animal = "ship"
hp5.target_animal = "plane"
hp5.pruning_dataset = "p_target"
hp5.save_opp = True

run_5 = Run('testing', 1, hp5, save_opp=hp5.save_opp)

runs = [run_5]

num_seeds = 3

target = True
naive = True

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(len(runs)):
        if target:
            run = runs[j]
        elif naive:
            run = Run.restore('exp_10', 2, num_seeds, naive=False, save_opp=True)
        else:
            run = Run.restore('exp_10', 2, num_seeds, save_opp=True)
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

    # save_results_to_bucket()