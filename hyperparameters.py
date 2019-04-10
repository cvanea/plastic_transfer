# TODO: See if keras has a built-in way to save model hyperparameters.
# TODO: Find a way to store the exact git commit hash.
import os
import numpy as np

class Hyperparameters:
    def __init__(self):
        self.csv_directory_name = None
        self.experiment_name = None
        self.num_starting_units = None
        self.lower_threshold = None
        self.upper_threshold = None
        self.source_lr = None
        self.target_lr = None
        self.batch_size = None
        self.conv_activation = None
        self.loss_function = None

    # @staticmethod
    # def from_csv(self, file):
    #     m = Hyperparameters()
    #     pd.read_csv(blah)
    #     m.

# Hyperparameters.from_csv(path)

    def to_csv(self):
        hyperparams = np.array(
            [str(self.num_starting_units), str(self.lower_threshold), str(self.upper_threshold), str(self.source_lr),
             str(self.target_lr), str(self.batch_size), self.conv_activation, self.loss_function])

        names = np.array(["num_starting_units", "lower_threshold", "upper_threshold", "source_lr", "target_lr",
                          "batch_size", "conv_activation", "loss_function"])

        # headers = "num_starting_units,lower_threshold,upper_threshold," \
        #           "source_lr,target_lr,batch_size,conv_activation,loss_function"

        file_dirs = "results/" + self.experiment_name
        file_name = "params.csv"

        if not os.path.isdir(file_dirs):
            os.makedirs(file_dirs)
        file_path = os.path.join(file_dirs, file_name)

        np.savetxt(file_path, np.column_stack((names, hyperparams)), delimiter=",", header=None, fmt="%s", comments=None)



def record_metadata(experiment_name, num_starting_units, lower_threshold, upper_threshold,
                    source_lr, target_lr, batch_size, conv_activation, loss_function):
    pass





# /results
#  /experimentID
#    params.csv
#    run1/
#      train/
#      seeded_units_stopped_epoch.csv
#        seeded/
#          mcc.csv
#          accuracy.csv
#        naive/
#          mcc.csv
#          accuracy.csv
#      val/
#        seeded/
#          mcc.csv
#          accuracy.csv
#        naive/
#          mcc.csv
#          accuracy.csv
#      test
#        seeded/
#          mcc.csv
#          accuracy.csv
#        naive/
#          mcc.csv
#          accuracy.csv