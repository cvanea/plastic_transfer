import matplotlib.pyplot as plt
import seaborn
seaborn.set()

from run import Run
from utils import create_path


__PNG_DPI__ = 150


def gen_graphs(run):
    for d in ("test", "train", "val"):
        naive_dataset = getattr(run.naive, d)
        seeded_dataset = getattr(run.seeded, d)
        for m in ("acc", "mcc"):
            n = getattr(naive_dataset, m).df
            s = getattr(seeded_dataset, m).df
            single_network_performance(f"naive {d}", m, naive_dataset.path, n)
            single_network_performance(f"seeded {d}", m, seeded_dataset.path, s)

            single_network_performance(f"naive average {d}", m, naive_dataset.path, n.mean(axis=1))
            single_network_performance(f"seeded average {d}", m, seeded_dataset.path, s.mean(axis=1))

            compare_network_performance(d, m, run.path, s, n)
            compare_average_network_performance(d, m, run.path, s.mean(axis=1), n.mean(axis=1))

    for n in ("naive", "seeded"):
        network = getattr(run, n)
        data = {"test": {}, "train": {}, "val": {}}
        for d in data.keys():
            dataset = getattr(network, d)
            for m in ("acc", "mcc"):
                measure = getattr(dataset, m).df
                data[d][m] = measure.mean(axis=1)

        all_averaged_dataset_performance(n, m, network.path, data)


# All seeds of the performance measure for one network
def single_network_performance(title, measure, path, seeded_data):
    plt.xlabel('epochs')
    plt.ylabel(measure)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.plot(seeded_data)
    # plt.show()
    plt.savefig(create_path(path, f"{measure}.png"), dpi=__PNG_DPI__)
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
    plt.savefig(create_path(path, f"compare_{dataset}_{measure}.png"), dpi=__PNG_DPI__)
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
    plt.savefig(create_path(path, f"compare_average_{dataset}_{measure}.png"), dpi=__PNG_DPI__)
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
    plt.savefig(create_path(path, f"all_datasets_{measure}.png"), dpi=__PNG_DPI__)
    plt.clf()


if __name__ == "__main__":
    r = Run.restore("testing", 0)
    gen_graphs(r)
