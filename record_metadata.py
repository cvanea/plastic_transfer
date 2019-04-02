# TODO: See if keras has a built-in way to save model hyperparameters.
# TODO: Find a way to store the exact git commit hash.
# TODO: Put into a format that can be read to recreate. json?
import os
import numpy as np


def record_metadata(csv_directory_name, experiment_number, num_starting_units, num_seeded_units,
                    lower_threshold, upper_threshold, source_stopped_epoch, source_lr, target_lr, batch_size,
                    conv_activation, loss_function):

    metadata = np.array([str(num_starting_units), str(num_seeded_units), str(lower_threshold), str(upper_threshold),
                         str(source_stopped_epoch), str(source_lr), str(target_lr), str(batch_size), conv_activation,
                         loss_function])

    names = np.array(["num_starting_units", "num_seeded_units", "lower_threshold", "upper_threshold", "source_stopped_epoch",
              "source_lr", "target_lr", "batch_size", "conv_activation", "loss_function"])

    # headers = "num_starting_units,num_seeded_units,lower_threshold,upper_threshold,source_stopped_epoch," \
    #           "source_lr,target_lr,batch_size,conv_activation,loss_function"

    file_dirs = "metadata/" + csv_directory_name
    file_name = experiment_number + ".csv"

    if not os.path.isdir(file_dirs):
        os.makedirs(file_dirs)
    file_path = os.path.join(file_dirs, file_name)

    np.savetxt(file_path, np.column_stack((names, metadata)), delimiter=",", header=None, fmt="%s", comments=None)

