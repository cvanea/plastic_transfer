from utils import create_path
import pandas as pd
from hyperparameters import Hyperparameters


class Run:
    def __init__(self, experiment_name, run_num, hyperparameters=None, save_opp=False):
        self.hyperparameters = hyperparameters
        self.path = create_path('results', experiment_name, "run_{}".format(str(run_num)))
        self.single_data = None
        self.apoz_data = None
        self.activation_data = {}
        self.target = NetworkType(create_path(self.path, 'target'), save_opp)
        self.naive = NetworkType(create_path(self.path, 'naive'), save_opp=False)
        self.source = NetworkType(create_path(self.path, 'source'), save_opp)

    def save(self, target=True, naive=True):
        self._save_hp()
        if target:
            if self.hyperparameters.save_opp:
                self.source._save()
            self.target._save()
            self._save_single_data()
            self._save_apoz_data()
            self._save_activation_data()
        if naive:
            self.naive._save()


    @staticmethod
    def restore(experiment_name, run_num, num_seeds, target=True, naive=True, save_opp=False):
        r = Run(experiment_name, run_num, save_opp=save_opp)
        r._restore_hp()
        r._restore_single_data()
        r._restore_apoz_data()
        r._restore_activation_data(num_seeds)
        if target:
            r.target._restore(r.hyperparameters.save_opp)
        if naive:
            r.naive._restore()
        if r.hyperparameters.save_opp:
            r.source._restore(r.hyperparameters.save_opp)
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
        path = create_path(self.path, 'activations')
        self.apoz_data.to_csv(create_path(path, "apoz_data.csv"))

    def _restore_apoz_data(self):
        path = create_path(self.path, 'activations')
        self.apoz_data = pd.read_csv(create_path(path, "apoz_data.csv"), index_col=0)

    def update_activation_data(self, seed, activations):
        self.activation_data[str(seed)] = pd.DataFrame(activations)

    def _save_activation_data(self):
        for key in self.activation_data:
            path = create_path(self.path, 'activations')
            self.activation_data[key].to_csv(create_path(path, "seed_{}_activations.csv".format(key)))

    def _restore_activation_data(self, num_seeds):
        path = create_path(self.path, 'activations')
        for seed in range(num_seeds):
            self.activation_data[str(seed)] = pd.read_csv(
                create_path(path, "seed_{}_activations.csv".format(str(seed))), index_col=0)


class NetworkType:
    def __init__(self, path, save_opp):
        self.path = path
        self.train = Dataset(create_path(path, 'train'), save_opp)
        self.val = Dataset(create_path(path, 'val'), save_opp)
        self.test = Dataset(create_path(path, 'test'), save_opp)

    def update(self, seed, predictions):
        self.train.update(seed, predictions.train)
        self.val.update(seed, predictions.val)
        self.test.update(seed, predictions.test)

    def _save(self):
        self.train._save()
        self.val._save()
        self.test._save()

    def _restore(self, save_opp=False):
        self.train._restore(save_opp)
        self.val._restore(save_opp)
        self.test._restore(save_opp)


class Dataset:
    def __init__(self, path, save_opp):
        self.path = path
        self.save_opp = save_opp
        self.mcc = Measure(create_path(path, 'mcc.csv'))
        self.acc = Measure(create_path(path, 'accuracy.csv'))
        if save_opp:
            self.opp_mcc = Measure(create_path(path, 'opp_mcc.csv'))

    def update(self, seed, predictions):
        self.mcc.update(seed, predictions['mcc'])
        self.acc.update(seed, predictions['acc'])
        if self.save_opp:
            self.opp_mcc.update(seed, predictions['opp_mcc'])

    def _save(self):
        self.mcc._save()
        self.acc._save()
        if self.save_opp:
            self.opp_mcc._save()

    def _restore(self, save_opp):
        self.mcc._restore()
        self.acc._restore()
        if save_opp:
            self.opp_mcc._restore()


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
