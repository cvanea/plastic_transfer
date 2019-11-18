import pandas as pd

import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import minmax_scale

seaborn.set()

from run import Run
from utils import create_path
import numpy as np

__PNG_DPI__ = 150


def gen_graphs(run, target=True, naive=True, save_opp=False):
    source_animal = run.hyperparameters.source_animal
    target_animal = run.hyperparameters.target_animal

    for d in ("test", "train", "val"):
        if not naive:
            target_dataset = getattr(run.target, d)
        else:
            naive_dataset = getattr(run.naive, d)
            target_dataset = getattr(run.target, d)
        for m in ("acc", "mcc"):
            if not naive:
                s = getattr(target_dataset, m).df
                single_network_performance("target {} {} to {}".format(d, source_animal, target_animal),
                                           m, target_dataset.path, s)
                single_network_performance("target average {} {} to {}".format(d, source_animal, target_animal),
                                           "{} average".format(m), target_dataset.path, s.mean(axis=1))
            elif not target:
                n = getattr(naive_dataset, m).df
                s = getattr(target_dataset, m).df
                single_network_performance("naive {} {}".format(d, target_animal), m, naive_dataset.path, n)
                single_network_performance("naive average {} {}".format(d, target_animal),
                                           "{} average".format(m), naive_dataset.path, n.mean(axis=1))
                compare_network_performance(d, m, run.path, s, n, source_animal, target_animal)
                compare_average_network_performance(d, m, run.path, s.mean(axis=1), n.mean(axis=1), "target", "naive",
                                                    source_animal, target_animal)
            elif target and naive:
                n = getattr(naive_dataset, m).df
                s = getattr(target_dataset, m).df
                single_network_performance("target {} {} to {}".format(d, source_animal, target_animal),
                                           m, target_dataset.path, s)
                single_network_performance("target average {} {} to {}".format(d, source_animal, target_animal),
                                           "{} average".format(m), target_dataset.path, s.mean(axis=1))
                single_network_performance("naive {} {}".format(d, target_animal), m,
                                           naive_dataset.path, n)
                single_network_performance("naive average {} {}".format(d, target_animal),
                                           "{} average".format(m), naive_dataset.path, n.mean(axis=1))
                compare_network_performance(d, m, run.path, s, n, source_animal, target_animal)
                compare_average_network_performance(d, m, run.path, s.mean(axis=1), n.mean(axis=1), "target", "naive",
                                                    source_animal, target_animal)

    if target:
        network = run.target
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("acc", "mcc", "opp_mcc"):
                if m == "opp_mcc" and not save_opp:
                    continue
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

            if save_opp:
                compare_average_network_performance(d, "mcc", network.path, data[d]["mcc"], data[d]["opp_mcc"],
                                                    "target", "source", source_animal, target_animal)

        all_averaged_dataset_performance("target", "acc", network.path, data, source_animal, target_animal)
        all_averaged_dataset_performance("target", "mcc", network.path, data, source_animal, target_animal)

        # activation_mean_histogram(run.path, run.activation_data)
        all_activation_mean_histogram(run.path, run.activation_data)

        # scatter_std_mean_activation(run.path, run.activation_data)
        all_scatter_std_mean_activation(run.path, run.activation_data)

        all_normalised_maru_histogram(run.path, run.activation_data)

    if naive:
        network = run.naive
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("acc", "mcc"):
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

        all_averaged_dataset_performance("naive", "acc", network.path, data, source_animal, target_animal)
        all_averaged_dataset_performance("naive", "mcc", network.path, data, source_animal, target_animal)

    if save_opp:
        network = run.source
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("mcc", 'opp_mcc'):
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

            compare_average_network_performance(d, "mcc", network.path, data[d]["mcc"], data[d]["opp_mcc"], "source",
                                                "target", source_animal, target_animal)


# All seeds of the performance measure for one network
def single_network_performance(title, measure, path, data):
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.plot(data)
    # plt.show()
    plt.savefig(create_path(path, "{}.png".format(measure)), dpi=__PNG_DPI__)
    plt.clf()


# Comparison of performance measure for two networks for one dataset for all seeds
def compare_network_performance(dataset, measure, path, first_data, second_data, source_animal, target_animal):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(first_data, label='target')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title("target " + dataset + " " + source_animal + " to " + target_animal)
    plt.ylim(0.0, 1.0)
    plt.subplot(212)
    plt.plot(second_data, label='naive')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title("naive " + dataset + " " + target_animal)
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "compare_{}_{}.png".format(dataset, measure)), dpi=__PNG_DPI__)
    plt.clf()


# Comparison of average performance measure for two networks for one dataset
def compare_average_network_performance(dataset, measure, path, first_data, second_data, first_label, second_label,
                                        source_animal, target_animal):
    if first_label == "target":
        plt.plot(first_data, label=first_label, color='C0')
        plt.plot(second_data, label=second_label, color='C1')
    else:
        plt.plot(first_data, label=first_label, color='C1')
        plt.plot(second_data, label=second_label, color='C0')

    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(dataset + ' ' + measure + " averaged" + " " + source_animal + " to " + target_animal)
    plt.legend(loc='lower right')
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "compare_average_{}_{}_{}.png".format(dataset, measure, second_label)),
                dpi=__PNG_DPI__)
    plt.clf()


# Averaged performance measure across all datasets for one network
def all_averaged_dataset_performance(network_name, measure, path, data, source_animal, target_animal):
    plt.plot(data["train"][measure], label='train')
    plt.plot(data["val"][measure], label='val')
    plt.plot(data["test"][measure], label='test')
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(network_name + " averaged results " + source_animal + " to " + target_animal)
    plt.legend(loc='lower right')
    plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(create_path(path, "all_datasets_{}.png".format(measure)), dpi=__PNG_DPI__)
    plt.clf()


def activation_mean_histogram(path, activation_data):
    path = create_path(path, "activations")
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data, axis=0)

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
        mean_data = np.mean(data, axis=0)
        activation_data_list.append(mean_data)

    n, bins, patches = plt.hist(activation_data_list, 20, range=(0.0, 1.0), label=activation_data.keys(), rwidth=1.0,
                                linewidth=0)
    plt.ylabel('number of neurons')
    plt.xlabel('mean activation')
    plt.title('mean activation for all seeds')
    # plt.legend(loc='upper right')
    # plt.ylim(0, 40)
    # plt.show()
    path = create_path(path, "activations")
    plt.savefig(create_path(path, "all_seeds_mean_act.png"), dpi=__PNG_DPI__)
    plt.clf()


def scatter_std_mean_activation(path, activation_data):
    path = create_path(path, "activations")
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)

        plt.scatter(mean_data, std_data)
        plt.xlabel('mean activation')
        plt.ylabel('std of activation')
        plt.title('mean vs std for seed {}'.format(seed))
        # plt.ylim(0.0, 4.0)
        # plt.xlim(0.0, 3.0)
        # plt.show()
        plt.savefig(create_path(path, "seed_{}_mean_std_act.png".format(seed)), dpi=__PNG_DPI__)
        plt.clf()


def all_scatter_std_mean_activation(path, activation_data):
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        plt.scatter(mean_data, std_data, label=seed)

    plt.xlabel('mean activation')
    plt.ylabel('std of activation')
    plt.title('mean vs std for all seeds')
    # plt.ylim(0.0, 4.0)
    # plt.xlim(0.0, 8.0)
    # plt.legend(loc='upper right')
    # plt.show()
    path = create_path(path, "activations")
    plt.savefig(create_path(path, "all_seed_mean_std_act.png"), dpi=__PNG_DPI__)
    plt.clf()


def all_normalised_maru_histogram(path, activation_data):
    normalised_maru_data_list = []
    for seed in activation_data:
        data = activation_data[seed]
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        maru_data = np.divide(mean_data, std_data, out=np.zeros_like(mean_data), where=std_data != 0)
        normalised_maru = minmax_scale(maru_data, feature_range=(0, 1))
        normalised_maru_data_list.append(normalised_maru)

    n, bins, patches = plt.hist(normalised_maru_data_list, 20, range=(0.0, 1.0), label=activation_data.keys(),
                                rwidth=1.0,
                                linewidth=0)
    plt.ylabel('number of neurons')
    plt.xlabel('maru value')
    plt.title('normalised maru value for all seeds')
    # plt.legend(loc='upper right')
    # plt.ylim(0, 60)
    # plt.show()
    path = create_path(path, "activations")
    plt.savefig(create_path(path, "all_normal_maru.png"), dpi=__PNG_DPI__)
    plt.clf()


if __name__ == "__main__":
    # r = Run.restore("testing", 0, 1)
    # gen_graphs(r)

    data = {}
    data['0'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_0_activations.csv', index_col=0)
    data['1'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_1_activations.csv', index_col=0)
    data['2'] = pd.read_csv('results_cloud/results/exp_7/run_1/seed_2_activations.csv', index_col=0)
    # all_activation_mean_histogram('results_cloud/results/exp_7/run_1/', data)
    # all_scatter_std_mean_activation('results_cloud/results/exp_7/run_1/', data)
    all_normalised_maru_histogram('results_cloud/results/exp_7/run_1/', data)
