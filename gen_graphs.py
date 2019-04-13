import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import create_path


# All seeds of the performance measure for one network
def single_network_performance(experiment_name, run_num, network_name, dataset_type, measure):
    file_dir = create_path('results', experiment_name, f"run_{str(run_num)}", network_name, dataset_type)
    csv = measure + ".csv"
    file_path = create_path(file_dir, csv)

    if not os.path.isfile(file_path):
        print("No csv file found at that path.")
    else:
        data = pd.read_csv(file_path, index_col=0)
        plt.xlabel('epochs')
        plt.ylabel(measure)
        plt.title(network_name + " " + dataset_type + " results")
        plt.ylim(0.0, 1.0)
        plt.plot(data)
        # plt.show()
        plt.savefig(create_path(file_dir, f"{measure}.png"))


# Comparison of performance measure for two networks for one dataset for all seeds
def compare_network_performance(experiment_name, run_num, dataset_type, measure):
    file_path_1 = "results/" + experiment_name + "/run_" + str(run_num) + "/seeded/" + dataset_type + "/" + \
                  measure + ".csv"
    file_path_2 = "results/" + experiment_name + "/run_" + str(run_num) + "/naive/" + dataset_type + "/" + \
                  measure + ".csv"

    if not os.path.isfile(file_path_1) and not os.path.isfile(file_path_2):
        print("No csv file found at that path.")
    else:
        data_1 = pd.read_csv(file_path_1, index_col=0)
        data_2 = pd.read_csv(file_path_2, index_col=0)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(data_1, label='seeded')
        plt.xlabel('epochs')
        plt.ylabel(measure)
        plt.title("seeded " + dataset_type + " results")
        plt.ylim(0.0, 1.0)
        plt.subplot(212)
        plt.plot(data_2, label='naive')
        plt.xlabel('epochs')
        plt.ylabel(measure)
        plt.title("naive " + dataset_type + " results")
        plt.ylim(0.0, 1.0)
        plt.show()


# Average performance measure for one network for one dataset
def average_network_performance(experiment_name, run_num, network_name, dataset_type, measure):
    file_path = "results/" + experiment_name + "/run_" + str(run_num) + "/" + network_name + "/" + dataset_type + \
                "/" + measure + ".csv"

    if not os.path.isfile(file_path):
        print("No csv file found at that path.")
    else:
        data = pd.read_csv(file_path, index_col=0)
        data_averaged = data.mean(axis=1)
        plt.xlabel('epochs')
        plt.ylabel("average " + measure)
        plt.title(network_name + " averaged " + dataset_type + " results")
        plt.ylim(0.0, 1.0)
        plt.plot(data_averaged)
        plt.show()


# Comparison of average performance measure for two networks for one dataset
def compare_average_network_performance(experiment_name, run_num, dataset_type, measure):
    file_path_1 = "results/" + experiment_name + "/run_" + str(run_num) + "/seeded/" + dataset_type + "/" + \
                  measure + ".csv"
    file_path_2 = "results/" + experiment_name + "/run_" + str(run_num) + "/naive/" + dataset_type + "/" + \
                  measure + ".csv"

    if not os.path.isfile(file_path_1) and not os.path.isfile(file_path_2):
        print("No csv file found at that path.")
    else:
        data_1 = pd.read_csv(file_path_1, index_col=0).mean(axis=1)
        data_2 = pd.read_csv(file_path_2, index_col=0).mean(axis=1)

        plt.plot(data_1, label='seeded')
        plt.plot(data_2, label='naive')
        plt.xlabel('epochs')
        plt.ylabel(measure)
        plt.title(dataset_type + "data averaged results")
        plt.legend(loc='lower right')
        plt.ylim(0.0, 1.0)
        plt.show()


# Averaged performance measure across all datasets for one network
def all_averaged_dataset_performance(experiment_name, run_num, network_name, measure):
    file_dir = create_path("results", experiment_name, f"run_{str(run_num)}", network_name)
    file_train = "train/" + measure + ".csv"
    file_val = "val/" + measure + ".csv"
    file_test = "test/" + measure + ".csv"

    file_path_train = create_path(file_dir, file_train)
    file_path_val = create_path(file_dir, file_val)
    file_path_test = create_path(file_dir, file_test)

    if not os.path.isfile(file_path_train) and not os.path.isfile(file_path_val) and not os.path.isfile(file_path_test):
        print("No csv file found at that path.")
    else:
        data_train = pd.read_csv(file_path_train, index_col=0).mean(axis=1)
        data_val = pd.read_csv(file_path_val, index_col=0).mean(axis=1)
        data_test = pd.read_csv(file_path_test, index_col=0).mean(axis=1)

        plt.plot(data_train, label='train')
        plt.plot(data_val, label='val')
        plt.plot(data_test, label='test')
        plt.xlabel('epochs')
        plt.ylabel(measure)
        plt.title(network_name + " averaged results")
        plt.legend(loc='lower right')
        plt.ylim(0.0, 1.0)
        # plt.show()
        plt.savefig(create_path(file_dir, f"{measure}.png"))


if __name__ == "__main__":
    # single_network_performance("testing", 0, "seeded", "test", "mcc")

    # compare_network_performance("testing", 0, "train", 'mcc')

    # average_network_performance("testing", 0, "seeded", "train", "mcc")

    # compare_average_network_performance("testing", 0, "train", 'mcc')

    all_averaged_dataset_performance("testing", 0, "seeded", "mcc")
