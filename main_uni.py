"""Script for automating runs on uni GPU."""

import transfer_networks, naive_network, hyperparameters as hyperparam
from run import Run
from gen_graphs import gen_graphs

hp = hyperparam.Hyperparameters()
hp.source_max_epochs = 15
hp.target_max_epochs = 300
hp.num_starting_units = 100
hp.upper_threshold = 100.0
hp.lower_threshold = -0.1
hp.source_lr = 0.001
hp.target_lr = 0.0001
hp.batch_size = 200
hp.conv_activation = 'relu'
hp.loss_function = 'binary_crossentropy'
hp.pruning_method = 'thresh_maru'
hp.source_animal = "dog"
hp.target_animal = "cat"
hp.pruning_dataset = "p_target"
hp.save_opp = True
hp.labels_per_category = 500
hp.reinit_weights = False

run_1 = Run('exp_1', 8, hp, save_opp=hp.save_opp)

hp2 = hyperparam.Hyperparameters()
hp2.source_max_epochs = 15
hp2.target_max_epochs = 300
hp2.num_starting_units = 100
hp2.upper_threshold = 100.0
hp2.lower_threshold = -0.1
hp2.source_lr = 0.001
hp2.target_lr = 0.0001
hp2.batch_size = 200
hp2.conv_activation = 'relu'
hp2.loss_function = 'binary_crossentropy'
hp2.pruning_method = 'thresh_maru'
hp2.source_animal = "deer"
hp2.target_animal = "dog"
hp2.pruning_dataset = "p_target"
hp2.save_opp = True
hp2.labels_per_category = 500
hp2.reinit_weights = False

run_2 = Run('exp_1', 9, hp2, save_opp=hp2.save_opp)

hp3 = hyperparam.Hyperparameters()
hp3.source_max_epochs = 15
hp3.target_max_epochs = 300
hp3.num_starting_units = 100
hp3.upper_threshold = 100.0
hp3.lower_threshold = -0.1
hp3.source_lr = 0.001
hp3.target_lr = 0.0001
hp3.batch_size = 200
hp3.conv_activation = 'relu'
hp3.loss_function = 'binary_crossentropy'
hp3.pruning_method = 'thresh_maru'
hp3.source_animal = "deer"
hp3.target_animal = "horse"
hp3.pruning_dataset = "p_target"
hp3.save_opp = True
hp3.labels_per_category = 500
hp3.reinit_weights = False

run_3 = Run('exp_1', 10, hp3, save_opp=hp3.save_opp)

hp4 = hyperparam.Hyperparameters()
hp4.source_max_epochs = 15
hp4.target_max_epochs = 300
hp4.num_starting_units = 100
hp4.upper_threshold = 100.0
hp4.lower_threshold = -0.1
hp4.source_lr = 0.001
hp4.target_lr = 0.0001
hp4.batch_size = 200
hp4.conv_activation = 'relu'
hp4.loss_function = 'binary_crossentropy'
hp4.pruning_method = 'thresh_maru'
hp4.source_animal = "deer"
hp4.target_animal = "ship"
hp4.pruning_dataset = "p_target"
hp4.save_opp = True
hp4.labels_per_category = 500
hp4.reinit_weights = False

run_4 = Run('exp_1', 11, hp4, save_opp=hp4.save_opp)

hp5 = hyperparam.Hyperparameters()
hp5.source_max_epochs = 15
hp5.target_max_epochs = 300
hp5.num_starting_units = 100
hp5.upper_threshold = 100.0
hp5.lower_threshold = -0.1
hp5.source_lr = 0.001
hp5.target_lr = 0.0001
hp5.batch_size = 200
hp5.conv_activation = 'relu'
hp5.loss_function = 'binary_crossentropy'
hp5.pruning_method = 'thresh_maru'
hp5.source_animal = "deer"
hp5.target_animal = "plane"
hp5.pruning_dataset = "p_target"
hp5.save_opp = True
hp5.labels_per_category = 500
hp5.reinit_weights = False

run_5 = Run('exp_1', 12, hp5, save_opp=hp5.save_opp)

hp6 = hyperparam.Hyperparameters()
hp6.source_max_epochs = 15
hp6.target_max_epochs = 300
hp6.num_starting_units = 100
hp6.upper_threshold = 100.0
hp6.lower_threshold = -0.1
hp6.source_lr = 0.001
hp6.target_lr = 0.0001
hp6.batch_size = 200
hp6.conv_activation = 'relu'
hp6.loss_function = 'binary_crossentropy'
hp6.pruning_method = 'thresh_maru'
hp6.source_animal = "deer"
hp6.target_animal = "car"
hp6.pruning_dataset = "p_target"
hp6.save_opp = True
hp6.labels_per_category = 500
hp6.reinit_weights = False

run_6 = Run('exp_1', 13, hp6, save_opp=hp6.save_opp)

hp7 = hyperparam.Hyperparameters()
hp7.source_max_epochs = 15
hp7.target_max_epochs = 300
hp7.num_starting_units = 100
hp7.upper_threshold = 100.0
hp7.lower_threshold = -0.1
hp7.source_lr = 0.001
hp7.target_lr = 0.0001
hp7.batch_size = 200
hp7.conv_activation = 'relu'
hp7.loss_function = 'binary_crossentropy'
hp7.pruning_method = 'thresh_maru'
hp7.source_animal = "deer"
hp7.target_animal = "truck"
hp7.pruning_dataset = "p_target"
hp7.save_opp = True
hp7.labels_per_category = 500
hp7.reinit_weights = False

run_7 = Run('exp_1', 14, hp7, save_opp=hp7.save_opp)

hp8 = hyperparam.Hyperparameters()
hp8.source_max_epochs = 15
hp8.target_max_epochs = 300
hp8.num_starting_units = 100
hp8.upper_threshold = 100.0
hp8.lower_threshold = -0.1
hp8.source_lr = 0.001
hp8.target_lr = 0.0001
hp8.batch_size = 200
hp8.conv_activation = 'relu'
hp8.loss_function = 'binary_crossentropy'
hp8.pruning_method = 'thresh_maru'
hp8.source_animal = "ship"
hp8.target_animal = "cat"
hp8.pruning_dataset = "p_target"
hp8.save_opp = True
hp8.labels_per_category = 500
hp8.reinit_weights = False

run_8 = Run('exp_1', 15, hp8, save_opp=hp8.save_opp)

hp9 = hyperparam.Hyperparameters()
hp9.source_max_epochs = 15
hp9.target_max_epochs = 300
hp9.num_starting_units = 100
hp9.upper_threshold = 100.0
hp9.lower_threshold = -0.1
hp9.source_lr = 0.001
hp9.target_lr = 0.0001
hp9.batch_size = 200
hp9.conv_activation = 'relu'
hp9.loss_function = 'binary_crossentropy'
hp9.pruning_method = 'thresh_maru'
hp9.source_animal = "ship"
hp9.target_animal = "dog"
hp9.pruning_dataset = "p_target"
hp9.save_opp = True
hp9.labels_per_category = 500
hp9.reinit_weights = False

run_9 = Run('exp_1', 16, hp9, save_opp=hp9.save_opp)

hp10 = hyperparam.Hyperparameters()
hp10.source_max_epochs = 15
hp10.target_max_epochs = 300
hp10.num_starting_units = 100
hp10.upper_threshold = 100.0
hp10.lower_threshold = -0.1
hp10.source_lr = 0.001
hp10.target_lr = 0.0001
hp10.batch_size = 200
hp10.conv_activation = 'relu'
hp10.loss_function = 'binary_crossentropy'
hp10.pruning_method = 'thresh_maru'
hp10.source_animal = "ship"
hp10.target_animal = "deer"
hp10.pruning_dataset = "p_target"
hp10.save_opp = True
hp10.labels_per_category = 500
hp10.reinit_weights = False

run_10 = Run('exp_1', 17, hp10, save_opp=hp10.save_opp)

hp11 = hyperparam.Hyperparameters()
hp11.source_max_epochs = 15
hp11.target_max_epochs = 300
hp11.num_starting_units = 100
hp11.upper_threshold = 100.0
hp11.lower_threshold = -0.1
hp11.source_lr = 0.001
hp11.target_lr = 0.0001
hp11.batch_size = 200
hp11.conv_activation = 'relu'
hp11.loss_function = 'binary_crossentropy'
hp11.pruning_method = 'thresh_maru'
hp11.source_animal = "ship"
hp11.target_animal = "horse"
hp11.pruning_dataset = "p_target"
hp11.save_opp = True
hp11.labels_per_category = 500
hp11.reinit_weights = False

run_11 = Run('exp_1', 18, hp11, save_opp=hp11.save_opp)

hp12 = hyperparam.Hyperparameters()
hp12.source_max_epochs = 15
hp12.target_max_epochs = 300
hp12.num_starting_units = 100
hp12.upper_threshold = 100.0
hp12.lower_threshold = -0.1
hp12.source_lr = 0.001
hp12.target_lr = 0.0001
hp12.batch_size = 200
hp12.conv_activation = 'relu'
hp12.loss_function = 'binary_crossentropy'
hp12.pruning_method = 'thresh_maru'
hp12.source_animal = "ship"
hp12.target_animal = "plane"
hp12.pruning_dataset = "p_target"
hp12.save_opp = True
hp12.labels_per_category = 500
hp12.reinit_weights = False

run_12 = Run('exp_1', 19, hp12, save_opp=hp12.save_opp)

hp13 = hyperparam.Hyperparameters()
hp13.source_max_epochs = 15
hp13.target_max_epochs = 300
hp13.num_starting_units = 100
hp13.upper_threshold = 100.0
hp13.lower_threshold = -0.1
hp13.source_lr = 0.001
hp13.target_lr = 0.0001
hp13.batch_size = 200
hp13.conv_activation = 'relu'
hp13.loss_function = 'binary_crossentropy'
hp13.pruning_method = 'thresh_maru'
hp13.source_animal = "ship"
hp13.target_animal = "car"
hp13.pruning_dataset = "p_target"
hp13.save_opp = True
hp13.labels_per_category = 500
hp13.reinit_weights = False

run_13 = Run('exp_1', 20, hp13, save_opp=hp13.save_opp)

hp14 = hyperparam.Hyperparameters()
hp14.source_max_epochs = 15
hp14.target_max_epochs = 300
hp14.num_starting_units = 100
hp14.upper_threshold = 100.0
hp14.lower_threshold = -0.1
hp14.source_lr = 0.001
hp14.target_lr = 0.0001
hp14.batch_size = 200
hp14.conv_activation = 'relu'
hp14.loss_function = 'binary_crossentropy'
hp14.pruning_method = 'thresh_maru'
hp14.source_animal = "ship"
hp14.target_animal = "truck"
hp14.pruning_dataset = "p_target"
hp14.save_opp = True
hp14.labels_per_category = 500
hp14.reinit_weights = False

run_14 = Run('exp_1', 21, hp14, save_opp=hp14.save_opp)

hp15 = hyperparam.Hyperparameters()
hp15.source_max_epochs = 15
hp15.target_max_epochs = 300
hp15.num_starting_units = 100
hp15.upper_threshold = 100.0
hp15.lower_threshold = -0.1
hp15.source_lr = 0.001
hp15.target_lr = 0.0001
hp15.batch_size = 200
hp15.conv_activation = 'relu'
hp15.loss_function = 'binary_crossentropy'
hp15.pruning_method = 'thresh_maru'
hp15.source_animal = "car"
hp15.target_animal = "cat"
hp15.pruning_dataset = "p_target"
hp15.save_opp = True
hp15.labels_per_category = 500
hp15.reinit_weights = False

run_15 = Run('exp_1', 22, hp15, save_opp=hp15.save_opp)

hp16 = hyperparam.Hyperparameters()
hp16.source_max_epochs = 15
hp16.target_max_epochs = 300
hp16.num_starting_units = 100
hp16.upper_threshold = 100.0
hp16.lower_threshold = -0.1
hp16.source_lr = 0.001
hp16.target_lr = 0.0001
hp16.batch_size = 200
hp16.conv_activation = 'relu'
hp16.loss_function = 'binary_crossentropy'
hp16.pruning_method = 'thresh_maru'
hp16.source_animal = "car"
hp16.target_animal = "dog"
hp16.pruning_dataset = "p_target"
hp16.save_opp = True
hp16.labels_per_category = 500
hp16.reinit_weights = False

run_16 = Run('exp_1', 23, hp16, save_opp=hp16.save_opp)

hp17 = hyperparam.Hyperparameters()
hp17.source_max_epochs = 15
hp17.target_max_epochs = 300
hp17.num_starting_units = 100
hp17.upper_threshold = 100.0
hp17.lower_threshold = -0.1
hp17.source_lr = 0.001
hp17.target_lr = 0.0001
hp17.batch_size = 200
hp17.conv_activation = 'relu'
hp17.loss_function = 'binary_crossentropy'
hp17.pruning_method = 'thresh_maru'
hp17.source_animal = "car"
hp17.target_animal = "deer"
hp17.pruning_dataset = "p_target"
hp17.save_opp = True
hp17.labels_per_category = 500
hp17.reinit_weights = False

run_17 = Run('exp_1', 24, hp17, save_opp=hp17.save_opp)

hp18 = hyperparam.Hyperparameters()
hp18.source_max_epochs = 15
hp18.target_max_epochs = 300
hp18.num_starting_units = 100
hp18.upper_threshold = 100.0
hp18.lower_threshold = -0.1
hp18.source_lr = 0.001
hp18.target_lr = 0.0001
hp18.batch_size = 200
hp18.conv_activation = 'relu'
hp18.loss_function = 'binary_crossentropy'
hp18.pruning_method = 'thresh_maru'
hp18.source_animal = "car"
hp18.target_animal = "horse"
hp18.pruning_dataset = "p_target"
hp18.save_opp = True
hp18.labels_per_category = 500
hp18.reinit_weights = False

run_18 = Run('exp_1', 25, hp18, save_opp=hp18.save_opp)

hp19 = hyperparam.Hyperparameters()
hp19.source_max_epochs = 15
hp19.target_max_epochs = 300
hp19.num_starting_units = 100
hp19.upper_threshold = 100.0
hp19.lower_threshold = -0.1
hp19.source_lr = 0.001
hp19.target_lr = 0.0001
hp19.batch_size = 200
hp19.conv_activation = 'relu'
hp19.loss_function = 'binary_crossentropy'
hp19.pruning_method = 'thresh_maru'
hp19.source_animal = "car"
hp19.target_animal = "ship"
hp19.pruning_dataset = "p_target"
hp19.save_opp = True
hp19.labels_per_category = 500
hp19.reinit_weights = False

run_19 = Run('exp_1', 26, hp19, save_opp=hp19.save_opp)

hp20 = hyperparam.Hyperparameters()
hp20.source_max_epochs = 15
hp20.target_max_epochs = 300
hp20.num_starting_units = 100
hp20.upper_threshold = 100.0
hp20.lower_threshold = -0.1
hp20.source_lr = 0.001
hp20.target_lr = 0.0001
hp20.batch_size = 200
hp20.conv_activation = 'relu'
hp20.loss_function = 'binary_crossentropy'
hp20.pruning_method = 'thresh_maru'
hp20.source_animal = "car"
hp20.target_animal = "plane"
hp20.pruning_dataset = "p_target"
hp20.save_opp = True
hp20.labels_per_category = 500
hp20.reinit_weights = False

run_20 = Run('exp_1', 27, hp20, save_opp=hp20.save_opp)

hp21 = hyperparam.Hyperparameters()
hp21.source_max_epochs = 15
hp21.target_max_epochs = 300
hp21.num_starting_units = 100
hp21.upper_threshold = 100.0
hp21.lower_threshold = -0.1
hp21.source_lr = 0.001
hp21.target_lr = 0.0001
hp21.batch_size = 200
hp21.conv_activation = 'relu'
hp21.loss_function = 'binary_crossentropy'
hp21.pruning_method = 'thresh_maru'
hp21.source_animal = "car"
hp21.target_animal = "truck"
hp21.pruning_dataset = "p_target"
hp21.save_opp = True
hp21.labels_per_category = 500
hp21.reinit_weights = False

run_21 = Run('exp_1', 28, hp21, save_opp=hp21.save_opp)


# runs = [run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9, run_10, run_11, run_12, run_13, run_14, run_15, run_16]
# runs = [run_17, run_18, run_19, run_20, run_21]
runs = [run_1]

num_seeds = 3

target = True
naive = True

if __name__ == "__main__":

    # Generate num_seeds number of weight initialisation runs.
    for j in range(len(runs)):
    # for j in range(8, 24):
        if target:
            run = runs[j]
        elif naive:
            run = Run.restore('exp_1', 1, num_seeds, naive=False, save_opp=True)
        else:
            run = Run.restore('exp_1', j, num_seeds, save_opp=True)
            gen_graphs(run, save_opp=True)

        if target or naive:
            for i in range(num_seeds):
                print("Current seed = " + str(i))

                # Calls source and target network function
                if target:
                    transfer_networks.network(i, run, run.hyperparameters)
                    run.save(naive=False)

                # Calls naive network function
                if naive:
                    seeded_units = run.single_data.loc['num_seeded_units', str(i)]
                    naive_network.network(i, run, run.hyperparameters, seeded_units)
                    run.save(target=False)

            gen_graphs(run, target=target, naive=naive, save_opp=run.hyperparameters.save_opp)
