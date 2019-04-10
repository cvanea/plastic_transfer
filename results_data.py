import os
import numpy as np
import pandas


class GenerateResults():
    def __init__(self, model, network_name, all_predictions, csv_directory_name, run_num, seed):
        self.model = model
        self.network_name = network_name
        self.all_predictions = all_predictions
        self.experiment_name = csv_directory_name
        self.run_num = run_num
        self.seed = seed

    # Generate training results
    def generate_train_data(self, data_name):
        if data_name == 'mcc':
            all_train_mcc = np.array(self.all_predictions.train_mcc)
            self._generate_dataset_csv('train', 'mcc', all_train_mcc)
        elif data_name == 'accuracy':
            all_train_accuracy = np.array(self.all_predictions.train_accuracy)
            self._generate_dataset_csv('train', 'accuracy', all_train_accuracy)

    # Generate validation results
    def generate_val_data(self, data_name):
        if data_name == 'mcc':
            all_val_mcc = np.array(self.all_predictions.val_mcc)
            self._generate_dataset_csv('val', 'mcc', all_val_mcc)
        elif data_name == 'accuracy':
            all_val_accuracy = np.array(self.all_predictions.val_accuracy)
            self._generate_dataset_csv('val', 'accuracy', all_val_accuracy)

    # Generate testing results
    def generate_test_data(self, data_name):
        if data_name == 'mcc':
            all_test_mcc = np.array(self.all_predictions.test_mcc)
            self._generate_dataset_csv('test', 'mcc', all_test_mcc)
        elif data_name == 'accuracy':
            all_test_accuracy = np.array(self.all_predictions.test_accuracy)
            self._generate_dataset_csv('test', 'accuracy', all_test_accuracy)

    # Record number of seeded units and the source stopped epoch
    def generate_seeded_units_stopped_epoch_data(self, num_seeded_units, source_stopped_epoch):
        d = {'seed': self.seed, 'num_seeded_units': num_seeded_units, 'source_stopped_epoch': source_stopped_epoch}

        self._generate_csv_single(pandas.DataFrame(data=d))

    def _generate_dataset_csv(self, dataset_type, data_name, data_array):
        data_array = pandas.DataFrame(data_array)

        file_dirs = "results/" + self.experiment_name + "/run" + self.run_num + "/" + dataset_type + "/" + \
                    self.network_name
        file_name = data_name + ".csv"

        if not os.path.isdir(file_dirs):
            os.makedirs(file_dirs)
        file_path = os.path.join(file_dirs, file_name)

        if not os.path.isfile(file_path):
            data_array.to_csv(file_path)
        else:
            prev_data = pandas.read_csv(file_path, index_col=0)
            prev_data[str(self.seed)] = data_array
            prev_data.to_csv(file_path)

    def _generate_csv_single(self, single_data):
        file_dirs = "results/" + self.experiment_name + "/run" + self.run_num
        file_name = "seeded_units_stopped_epoch.csv"

        if not os.path.isdir(file_dirs):
            os.makedirs(file_dirs)
        file_path = os.path.join(file_dirs, file_name)

        if not os.path.isfile(file_path):
            single_data.to_csv(file_path, index=None)
        else:
            prev_data = pandas.read_csv(file_path)
            prev_data.append(single_data, ignore_index=True)
            prev_data.to_csv(file_path, index=None)
