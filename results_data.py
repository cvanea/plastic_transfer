import os
import numpy as np
from sklearn.metrics import matthews_corrcoef


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
        # all_predictions.val_confusion.append(confusion_matrix(dog_val_labels, pred_val_classes).ravel())
        self.all_predictions.val_MCC.append(matthews_corrcoef(labels, pred_val_classes))

        # all_confusion_array = np.array(all_predictions.val_confusion)
        all_val_MCC = np.array(self.all_predictions.val_MCC)

        self._generate_csv('val', all_val_MCC)

    # Generate testing confusion history csv
    def generate_test_data(self, images, labels):
        pred_test_classes = self.model.predict_classes(images)
        # all_predictions.test_confusion_history.append(confusion_matrix(dog_test_labels, pred_test_classes).ravel())
        self.all_predictions.test_MCC.append(matthews_corrcoef(labels, pred_test_classes))

        # all_balanced_confusion_array = np.array(all_predictions.test_confusion_history)
        all_test_MCC = np.array(self.all_predictions.test_MCC)

        self._generate_csv('test', all_test_MCC)


    def _generate_csv(self, dataset_type, all_MCCs):
        if not os.path.isfile(
                "csv/" + self.csv_directory_name + "/" + dataset_type + "MCC/" + self.network_name + "_" + self.experiment_number + str(
                    self.seed) + ".csv"):
            np.savetxt(
                "csv/" + self.csv_directory_name + "/" + dataset_type + "MCC/" + self.network_name + "_" + self.experiment_number + str(
                    self.seed) + ".csv", np.column_stack(all_MCCs), delimiter=",", fmt="%s")
        else:
            with open(
                    "csv/" + self.csv_directory_name + "/" + dataset_type + "MCC/" + self.network_name + "_" + self.experiment_number + str(
                        self.seed) + ".csv",
                    "a") as f:
                np.savetxt(f, np.column_stack(all_MCCs), delimiter=",", fmt="%s")
