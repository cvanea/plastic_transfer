# TODO: Turn into class or other datastructure to set shared params.

import os
import numpy as np
from sklearn.metrics import matthews_corrcoef


# Generate training MCC history csv
def generate_train_data(model, network_name, images, labels, all_predictions, csv_directory_name, experiment_number,
                        seed):
    pred_train_classes = model.predict_classes(images)
    all_predictions.train_MCC.append(matthews_corrcoef(labels, pred_train_classes))

    all_train_MCC = np.array(all_predictions.train_MCC)

    _generate_csv('train', all_train_MCC, csv_directory_name, network_name, experiment_number, seed)


# Generate validation confusion history csv
def generate_val_data(model, network_name, images, labels, all_predictions, csv_directory_name, experiment_number,
                      seed):
    pred_val_classes = model.predict_classes(images)
    # all_predictions.val_confusion.append(confusion_matrix(dog_val_labels, pred_val_classes).ravel())
    all_predictions.val_MCC.append(matthews_corrcoef(labels, pred_val_classes))

    # all_confusion_array = np.array(all_predictions.val_confusion)
    all_val_MCC = np.array(all_predictions.val_MCC)

    _generate_csv('val', all_val_MCC, csv_directory_name, network_name, experiment_number, seed)


# Generate testing confusion history csv
def generate_test_data(model, network_name, images, labels, all_predictions, csv_directory_name, experiment_number,
                       seed):
    pred_test_classes = model.predict_classes(images)
    # all_predictions.test_confusion_history.append(confusion_matrix(dog_test_labels, pred_test_classes).ravel())
    all_predictions.test_MCC.append(matthews_corrcoef(labels, pred_test_classes))

    # all_balanced_confusion_array = np.array(all_predictions.test_confusion_history)
    all_test_MCC = np.array(all_predictions.test_MCC)

    _generate_csv('test', all_test_MCC, csv_directory_name, network_name, experiment_number, seed)


def _generate_csv(dataset_type, all_MCCs, csv_directory_name, network_name, experiment_number, seed):
    if not os.path.isfile(
            "csv/" + csv_directory_name + "/" + dataset_type + "MCC/" + network_name + "_" + experiment_number + str(
                seed) + ".csv"):
        np.savetxt(
            "csv/" + csv_directory_name + "/" + dataset_type + "MCC/" + network_name + "_" + experiment_number + str(
                seed) + ".csv", np.column_stack(all_MCCs), delimiter=",", fmt="%s")
    else:
        with open(
                "csv/" + csv_directory_name + "/" + dataset_type + "MCC/" + network_name + "_" + experiment_number + str(
                    seed) + ".csv",
                "a") as f:
            np.savetxt(f, np.column_stack(all_MCCs), delimiter=",", fmt="%s")
