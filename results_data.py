import os
import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas


class GenerateResults():
    def __init__(self, model, network_name, all_predictions, csv_directory_name, experiment_number, seed):
        self.model = model
        self.network_name = network_name
        self.all_predictions = all_predictions
        self.csv_directory_name = csv_directory_name
        self.experiment_number = experiment_number
        self.seed = seed

    # Generate training MCC history csv
    def generate_train_data(self, images, labels):
        pred_train_classes = self.model.predict_classes(images)
        self.all_predictions.train_MCC.append(matthews_corrcoef(labels, pred_train_classes))

        all_train_MCC = np.array(self.all_predictions.train_MCC)

        self._generate_csv('train', all_train_MCC)

    # Generate validation confusion history csv
    def generate_val_data(self, images, labels):
        pred_val_classes = self.model.predict_classes(images)
        self.all_predictions.val_MCC.append(matthews_corrcoef(labels, pred_val_classes))
        all_val_MCC = np.array(self.all_predictions.val_MCC)

        self._generate_csv('val', all_val_MCC)

    # Generate testing confusion history csv
    def generate_test_data(self, images, labels):
        pred_test_classes = self.model.predict_classes(images)
        self.all_predictions.test_MCC.append(matthews_corrcoef(labels, pred_test_classes))
        all_test_MCC = np.array(self.all_predictions.test_MCC)

        self._generate_csv('test', all_test_MCC)

    def _generate_csv(self, dataset_type, new_MCCs):
        new_MCCs = pandas.DataFrame(new_MCCs)

        file_dirs = "csv/" + self.csv_directory_name + "/" + dataset_type + "MCC"
        file_name = self.network_name + "_" + self.experiment_number + ".csv"

        if not os.path.isdir(file_dirs):
            os.makedirs(file_dirs)
        file_path = os.path.join(file_dirs, file_name)

        if not os.path.isfile(file_path):
            new_MCCs.to_csv(file_path)
        else:
            prev_MCCs = pandas.read_csv(file_path, index_col=0)
            prev_MCCs[str(self.seed)] = new_MCCs
            prev_MCCs.to_csv(file_path)
