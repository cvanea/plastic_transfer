import pandas as pd

import matplotlib.pyplot as plt
import seaborn

seaborn.set()

from run import Run
from utils import create_path
import numpy as np

__PNG_DPI__ = 150


def gen_graphs(run, seeded=True, naive=True):
    for d in ("test", "train", "val"):
        if not naive:
            seeded_dataset = getattr(run.seeded, d)
        else:
            naive_dataset = getattr(run.naive, d)
            seeded_dataset = getattr(run.seeded, d)
        for m in ("acc", "mcc"):
            if not naive:
                s = getattr(seeded_dataset, m).df
                single_network_performance("seeded {}".format(d), m, seeded_dataset.path, s)
                single_network_performance("seeded average {}".format(d), m, seeded_dataset.path, s.mean(axis=1))
            elif not seeded:
                n = getattr(naive_dataset, m).df
                s = getattr(seeded_dataset, m).df
                single_network_performance("naive {}".format(d), m, naive_dataset.path, n)
                single_network_performance("naive average {}".format(d), "{} average".format(m), naive_dataset.path,
                                           n.mean(axis=1))
                compare_network_performance(d, m, run.path, s, n)
                compare_average_network_performance(d, m, run.path, s.mean(axis=1), n.mean(axis=1))
            elif seeded and naive:
                n = getattr(naive_dataset, m).df
                s = getattr(seeded_dataset, m).df
                single_network_performance("seeded {}".format(d), m, seeded_dataset.path, s)
                single_network_performance("seeded average {}".format(d), "{} average".format(m), seeded_dataset.path,
                                           s.mean(axis=1))
                single_network_performance("naive {}".format(d), m, naive_dataset.path, n)
                single_network_performance("naive average {}".format(d), m, naive_dataset.path, n.mean(axis=1))
                compare_network_performance(d, m, run.path, s, n)
                compare_average_network_performance(d, m, run.path, s.mean(axis=1), n.mean(axis=1))

    if seeded:
        network = run.seeded
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("acc", "mcc"):
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

        all_averaged_dataset_performance("seeded", "acc", network.path, data)
        all_averaged_dataset_performance("seeded", "mcc", network.path, data)

        activation_mean_histogram(run.path, run.activation_data)
        all_activation_mean_histogram(run.path, run.activation_data)

    if naive:
        network = run.naive
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("acc", "mcc"):
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

        all_averaged_dataset_performance("naive", "acc", network.path, data)
        all_averaged_dataset_performance("naive", "mcc", network.path, data)


# All seeds of the performance measure for one network
def single_network_performance(title, measure, path, seeded_data):
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.plot(seeded_data)
    # plt.show()
    plt.savefig(create_path(path, "{}.png".format(measure)), dpi=__PNG_DPI__)
    plt.clf()


# Comparison of performance measure for two networks for one dataset for all seeds
def compare_network_performance(dataset, measure, path, seeded_data, naive_data):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(seeded_data, label='seeded')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title("seeded " + dataset)
    plt.ylim(0.0, 1.0)
    plt.subplot(212)
    plt.plot(naive_data, label='naive')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title("naive " + dataset)
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "compare_{}_{}.png".format(dataset, measure)), dpi=__PNG_DPI__)
    plt.clf()


# Comparison of average performance measure for two networks for one dataset
def compare_average_network_performance(dataset, measure, path, seeded_data, naive_data):
    plt.plot(seeded_data, label='seeded')
    plt.plot(naive_data, label='naive')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(dataset + ' ' + measure + " averaged")
    plt.legend(loc='lower right')
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "compare_average_{}_{}.png".format(dataset, measure)), dpi=__PNG_DPI__)
    plt.clf()


# Averaged performance measure across all datasets for one network
def all_averaged_dataset_performance(network_name, measure, path, data):
    plt.plot(data["train"][measure], label='train')
    plt.plot(data["val"][measure], label='val')
    plt.plot(data["test"][measure], label='test')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(network_name + " averaged results")
    plt.legend(loc='lower right')
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "all_datasets_{}.png".format(measure)), dpi=__PNG_DPI__)
    plt.clf()


def activation_mean_histogram(path, activation_data):
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data.values, axis=0)

        n, bins, patches = plt.hist(mean_data, 20, range=(0.0, 1.0))
        plt.ylabel('number of neurons')
        plt.xlabel('mean activation')
        plt.title('mean activation for seed {}'.format(seed))
        plt.ylim(0, 60)
        # plt.show()
        plt.savefig(create_path(path, "seed_{}_mean_act.png".format(seed)), dpi=__PNG_DPI__)
        plt.clf()


def all_activation_mean_histogram(path, activation_data):
    activation_data_list = []
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data.values, axis=0)
        activation_data_list.append(mean_data)

    n, bins, patches = plt.hist(activation_data_list, 20, range=(0.0, 1.0), label=activation_data.keys(), rwidth=1.0,
                                linewidth=0)
    plt.ylabel('number of neurons')
    plt.xlabel('mean activation')
    plt.title('mean activation for all seeds')
    plt.legend(loc='upper right')
    plt.ylim(0, 60)
    # plt.show()
    plt.savefig(create_path(path, "all_seeds_mean_act.png"), dpi=__PNG_DPI__)
    plt.clf()


if __name__ == "__main__":
    # r = Run.restore("testing", 0, 1)
    # gen_graphs(r)

    data = {}
    data['0'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_0_activations.csv', index_col=0)
    data['1'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_1_activations.csv', index_col=0)
    data['2'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_2_activations.csv', index_col=0)
    all_activation_mean_histogram('results_cloud/results/exp_7/run_1/seed_0_mean_act.png', data)
