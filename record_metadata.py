# TODO: See if keras has a built-in way to save model hyperparameters.
# TODO: Find a way to store the exact git commit hash.
# TODO: Put into a format that can be read to recreate. json?


def record_metadata(csv_directory_name, experiment_number, seed, num_starting_units, num_seeded_units,
                    lower_threshold, upper_threshold, source_stopped_epoch, source_lr, target_lr, batch_size,
                    conv_activation, loss_function):
    pass