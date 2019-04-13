# TODO: See if keras has a built-in way to save model hyperparameters.
# TODO: Find a way to store the exact git commit hash.
import os
import pandas as pd


class Hyperparameters:
    def __init__(self):
        self.experiment_name = None
        self.source_max_epochs = None
        self.target_max_epochs = None
        self.num_starting_units = None
        self.upper_threshold = None
        self.lower_threshold = None
        self.source_lr = None
        self.target_lr = None
        self.batch_size = None
        self.conv_activation = None
        self.loss_function = None

    def to_csv(self, run_num):
        d = {"source_max_epochs": [self.source_max_epochs], "target_max_epochs": [self.target_max_epochs],
             "num_starting_units": [self.num_starting_units], "upper_threshold": [self.upper_threshold],
             "lower_threshold": [self.lower_threshold], "source_lr": [self.source_lr], "target_lr": [self.target_lr],
             "batch_size": [self.batch_size], "conv_activation": [self.conv_activation],
             "loss_function": [self.loss_function]}
        hyperparams = pd.DataFrame(data=d)

        file_dirs = "results/" + self.experiment_name + "/run_" + str(run_num)
        file_name = "params.csv"

        if not os.path.isdir(file_dirs):
            os.makedirs(file_dirs)
        file_path = os.path.join(file_dirs, file_name)

        hyperparams.to_csv(file_path, index=None)


    # @staticmethod
    # def from_csv(self, file):
    #     m = Hyperparameters()
    #     pd.read_csv(blah)
    #     m.

    # Hyperparameters.from_csv(path)