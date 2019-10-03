import pandas as pd


def results_table(exp_name, starting_run, ending_run, based_on_string):
    all_table_data = pd.DataFrame()

    for run in range(starting_run, ending_run + 1):
        target_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/target/test/mcc.csv".format(exp_name, run),
                                      index_col=0)
        all_target_mcc_std = target_mcc_data.std(axis=1)
        target_mcc_std = all_target_mcc_std.mean()
        target_mcc_avg = target_mcc_data.mean(axis=1)
        max_value = max(target_mcc_avg)
        max_value_epoch = target_mcc_avg.argmax()
        start_mcc_value = target_mcc_avg[0]
        target_mcc_end = target_mcc_avg.iloc[-1]

        naive_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/naive/test/mcc.csv".format(exp_name, run),
                                     index_col=0)
        all_naive_mcc_std = naive_mcc_data.std(axis=1)
        naive_mcc_std = all_naive_mcc_std.mean()
        naive_mcc_avg = naive_mcc_data.mean(axis=1)
        naive_max_value = max(naive_mcc_avg)
        naive_max_value_epoch = naive_mcc_avg.argmax()
        naive_start_mcc_value = naive_mcc_avg[0]
        naive_mcc_end = naive_mcc_avg.iloc[-1]

        mcc_value_difference = max_value - naive_max_value
        end_difference = target_mcc_end - naive_mcc_end
        target_surpass_epoch = target_mcc_avg[target_mcc_avg > naive_max_value]

        if len(target_surpass_epoch) == 0:
            target_surpass_epoch = 0
        else:
            target_surpass_epoch = target_surpass_epoch.index[0]
        if all_table_data.empty:
            all_table_data = pd.DataFrame(
                [max_value, naive_max_value, mcc_value_difference, target_mcc_end, naive_mcc_end, end_difference,
                 target_mcc_std, naive_mcc_std, start_mcc_value, naive_start_mcc_value, max_value_epoch,
                 naive_max_value_epoch, target_surpass_epoch], index=[
                    "max target mcc", "max naive mcc", "mcc difference", "target end mcc", "naive end mxx",
                    "end difference", "target std", "naive std", "target start mcc", "naive start mcc",
                    "target max epoch", "naive max epoch", "target_surpass_epoch"], columns=[run])
        else:
            all_table_data[run] = [max_value, naive_max_value, mcc_value_difference, target_mcc_end, naive_mcc_end,
                                   end_difference, target_mcc_std, naive_mcc_std, start_mcc_value,
                                   naive_start_mcc_value, max_value_epoch, naive_max_value_epoch, target_surpass_epoch]

    all_table_data.to_csv("results_cloud/results/{}/{}.csv".format(exp_name, based_on_string))

    print(all_table_data)


def value_at_epoch(exp_name, num_runs):
    epoch_data = pd.read_csv("results_cloud/results/{}/peak_epochs.csv".format(exp_name), index_col=0)

    all_values = pd.DataFrame()

    target_test_values = []
    target_val_values = []
    naive_test_values = []
    naive_val_values = []

    for run in range(1, num_runs + 1):
        target_test_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/target/test/mcc.csv".format(exp_name, run),
                                           index_col=0)
        target_test_avg_mcc_data = target_test_mcc_data.mean(axis=1)

        target_val_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/target/val/mcc.csv".format(exp_name, run),
                                          index_col=0)
        target_val_avg_mcc_data = target_val_mcc_data.mean(axis=1)

        naive_test_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/naive/test/mcc.csv".format(exp_name, run),
                                          index_col=0)
        naive_test_avg_mcc_data = naive_test_mcc_data.mean(axis=1)

        naive_val_mcc_data = pd.read_csv("results_cloud/results/{}/run_{}/naive/val/mcc.csv".format(exp_name, run),
                                         index_col=0)
        naive_val_avg_mcc_data = naive_val_mcc_data.mean(axis=1)

        target_test_values.append(target_test_avg_mcc_data[epoch_data["target"][run]])
        target_val_values.append(target_val_avg_mcc_data[epoch_data["target"][run]])
        naive_test_values.append(naive_test_avg_mcc_data[epoch_data["naive"][run]])
        naive_val_values.append(naive_val_avg_mcc_data[epoch_data["naive"][run]])

    all_values["target_test"] = target_test_values
    all_values["target_val"] = target_val_values
    all_values["naive_test"] = naive_test_values
    all_values["naive_val"] = naive_val_values

    all_values.to_csv("results_cloud/results/{}/all_epoch_values.csv".format(exp_name))


if __name__ == "__main__":
    # results_table("exp_15", 1, 4, "based_on_cats")
    # results_table("exp_15", 5, 8, "based_on_horses")
    # results_table("exp_15", 9, 12, "based_on_ships")
    # results_table("exp_15", 13, 16, "based_on_trucks")
    # results_table("exp_15", 29, 35, "based_on_dogs")
    # results_table("exp_15", 36, 42, "based_on_deer")
    # results_table("exp_15", 43, 49, "based_on_planes")
    # results_table("exp_15", 50, 56, "based_on_cars")

    value_at_epoch("exp_15", 56)
