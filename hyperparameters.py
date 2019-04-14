# TODO: See if keras has a built-in way to save model hyperparameters.
# TODO: Find a way to store the exact git commit hash.
import pandas as pd
from utils import create_path


class Hyperparameters:
    def __init__(self):
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

    def to_csv(self, path):
        d = {"source_max_epochs": [self.source_max_epochs], "target_max_epochs": [self.target_max_epochs],
             "num_starting_units": [self.num_starting_units], "upper_threshold": [self.upper_threshold],
             "lower_threshold": [self.lower_threshold], "source_lr": [self.source_lr], "target_lr": [self.target_lr],
             "batch_size": [self.batch_size], "conv_activation": [self.conv_activation],
             "loss_function": [self.loss_function]}
        hyperparams = pd.DataFrame(data=d)

        file_path = create_path(path, 'params.csv')

        hyperparams.to_csv(file_path, index=None)

    @staticmethod
    def from_csv(file_path):
        m = Hyperparameters()
        hp_data = pd.read_csv(create_path(file_path, 'params.csv'))
        dict_hp_data = hp_data.to_dict(orient='rows')[0]

        m.source_max_epochs = dict_hp_data["source_max_epochs"]
        m.target_max_epochs = dict_hp_data["target_max_epochs"]
        m.num_starting_units = dict_hp_data["num_starting_units"]
        m.upper_threshold = dict_hp_data["upper_threshold"]
        m.lower_threshold = dict_hp_data["lower_threshold"]
        m.source_lr = dict_hp_data["source_lr"]
        m.target_lr = dict_hp_data["target_lr"]
        m.batch_size = dict_hp_data["batch_size"]
        m.conv_activation = dict_hp_data["conv_activation"]
        m.loss_function = dict_hp_data["loss_function"]

        return m
