from utils import create_path
import pandas as pd
from hyperparameters import Hyperparameters


class Run:
    def __init__(self, experiment_name, run_num, hyperparameters=None):
        self.hyperparameters = hyperparameters
        self.path = create_path('results', experiment_name, "run_{}".format(str(run_num)))
        self.single_data = None
        self.apoz_data = None
        self.activation_data = {}
        self.seeded = NetworkType(create_path(self.path, 'seeded'))
        self.naive = NetworkType(create_path(self.path, 'naive'))

    def save(self, seeded=True, naive=True):
        self._save_hp()
        if seeded:
            self.seeded._save()
            self._save_single_data()
            self._save_apoz_data()
            self._save_activation_data()
        if naive:
            self.naive._save()

    @staticmethod
    def restore(experiment_name, run_num, num_seeds, seeded=True, naive=True):
        r = Run(experiment_name, run_num)
        r._restore_hp()
        r._restore_single_data()
        r._restore_apoz_data()
        r._restore_activation_data(num_seeds)
        if seeded:
            r.seeded._restore()
        if naive:
            r.naive._restore()
        return r

    def _save_hp(self):
        self.hyperparameters.to_csv(self.path)

    def _restore_hp(self):
        self.hyperparameters = Hyperparameters.from_csv(self.path)

    def update_single_data(self, seed, num_seeded_units, source_stopped_epoch):
        if self.single_data is None:
            index = ["num_seeded_units", "source_stopped_epoch"]
            self.single_data = pd.DataFrame([num_seeded_units, source_stopped_epoch], index=index, columns=[str(seed)])
        else:
            self.single_data[str(seed)] = [num_seeded_units, source_stopped_epoch]

    def _save_single_data(self):
        self.single_data.to_csv(create_path(self.path, "seeded_units_stopped_epoch.csv"))

    def _restore_single_data(self):
        self.single_data = pd.read_csv(create_path(self.path, "seeded_units_stopped_epoch.csv"), index_col=0)

    def update_apoz_data(self, seed, apoz_data):
        if self.apoz_data is None:
            self.apoz_data = pd.DataFrame(apoz_data, columns=[str(seed)])
        else:
            self.apoz_data[str(seed)] = apoz_data

    def _save_apoz_data(self):
        self.apoz_data.to_csv(create_path(self.path, "apoz_data.csv"))

    def _restore_apoz_data(self):
        self.apoz_data = pd.read_csv(create_path(self.path, "apoz_data.csv"), index_col=0)

    def update_activation_data(self, seed, activations):
        self.activation_data[str(seed)] = pd.DataFrame(activations)

    def _save_activation_data(self):
        for key in self.activation_data:
            self.activation_data[key].to_csv(create_path(self.path, "seed_{}_activations.csv".format(key)))

    def _restore_activation_data(self, num_seeds):
        for seed in range(num_seeds):
            self.activation_data[str(seed)] = pd.read_csv(
                create_path(self.path, "seed_{}_activations.csv".format(str(seed))), index_col=0)


class NetworkType:
    def __init__(self, path):
        self.path = path
        self.train = Dataset(create_path(path, 'train'))
        self.val = Dataset(create_path(path, 'val'))
        self.test = Dataset(create_path(path, 'test'))

    def update(self, seed, predictions):
        self.train.update(seed, predictions.train)
        self.val.update(seed, predictions.val)
        self.test.update(seed, predictions.test)

    def _save(self):
        self.train._save()
        self.val._save()
        self.test._save()

    def _restore(self):
        self.train._restore()
        self.val._restore()
        self.test._restore()


class Dataset:
    def __init__(self, path):
        self.path = path
        self.mcc = Measure(create_path(path, 'mcc.csv'))
        self.acc = Measure(create_path(path, 'accuracy.csv'))

    def update(self, seed, predictions):
        self.mcc.update(seed, predictions['mcc'])
        self.acc.update(seed, predictions['acc'])

    def _save(self):
        self.mcc._save()
        self.acc._save()

    def _restore(self):
        self.mcc._restore()
        self.acc._restore()


class Measure:
    def __init__(self, path):
        self.path = path
        self.df = None

    def update(self, seed, data_array):
        if self.df is None:
            self.df = pd.DataFrame(data_array, columns=[str(seed)])
        else:
            self.df[str(seed)] = data_array

    def _save(self):
        self.df.to_csv(self.path)

    def _restore(self):
        self.df = pd.read_csv(self.path, index_col=0)
