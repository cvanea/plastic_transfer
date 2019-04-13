from utils import create_path
import pandas as pd


class Run:
    def __init__(self, network_name, csv_directory_name, run_num, all_predictions, hyperparameters):
        self.network_name = network_name
        self.experiment_name = csv_directory_name
        self.run_num = run_num
        self.all_predictions = all_predictions
        self.hyperparameters = hyperparameters
        self.single_data = None
        path = create_path('results', csv_directory_name, run_num)
        self.seeded = NetworkType(create_path(path, 'seeded'))
        self.naive = NetworkType(create_path(path, 'naive'))

    def save(self):
        self._save_results()
        self._save_hp()

    @staticmethod
    def restore(self, run_id):
        pass

    def _save_hp(self):
        self.hyperparameters.to_csv(self.run_num)

    def _save_results(self):
        self.seeded._save()
        self.naive._save()


class NetworkType:
    def __init__(self, path):
        self.train = Dataset(create_path(path, 'train'))
        self.val = Dataset(create_path(path, 'val'))
        self.test = Dataset(create_path(path, 'test'))

    # predictions.train.acc
    def update(self, seed, predictions):
        self.train.update(seed, predictions.train)
        self.val.update(seed, predictions.val)
        self.test.update(seed, predictions.test)

    def _save(self):
        self.train._save()
        self.val._save()
        self.test._save()

    # def _restore(self):
    #     self.mcc._restore()
    #     self.acc._restore()


class Dataset:
    def __init__(self, path):
        self.mcc = Measure(create_path(path, 'mcc.csv'))
        self.acc = Measure(create_path(path, 'acc.csv'))

    def update(self, seed, predictions):
        self.mcc.update(seed, predictions['mcc'])
        self.acc.update(seed, predictions['acc'])

    def _save(self):
        self.mcc._save()
        self.acc._save()

    # def _restore(self):
    #     self.mcc._restore()
    #     self.acc._restore()


class Measure:
    def __init__(self, path):
        self.path = path
        self.df = None

    def update(self, seed, data_array):
        if self.df is None:
            self.df = pd.DataFrame(data_array)
        else:
            self.df[str(seed)] = pd.Series(data_array) # Check if saving the seed as header

    def _save(self):
        self.df.to_csv(self.path)

    def _restore(self):
        pass




# run.seeded.train.mcc.update(seed, data)
