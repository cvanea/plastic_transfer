"""Script for aggregating MCC values."""

import pandas
import os
import math

# Csvs from which to extract MCC values.
csv0 = "GPU_csvs/GPUConfusionData/Units/1.0.csv"
csv1 = "GPU_csvs/GPUConfusionData/Units/1.1.csv"
csv2 = "GPU_csvs/GPUConfusionData/Units/1.2.csv"
csv3 = "GPU_csvs/GPUConfusionData/Units/1.3.csv"
csv4 = "GPU_csvs/GPUConfusionData/Units/1.4.csv"

# Csv with aggregated csv values
final_csv = "GPU_csvs/GPUConfusionData/Units/MCC/MCC1.csv"


def generate_MCC_csv(csv0, csv1, csv2, csv3, csv4, final_csv):

    col_names = ['tn', 'fp', 'fn', 'tp', 'CK', 'MCC']
    data0 = pandas.read_csv(csv0, names=col_names)
    data1 = pandas.read_csv(csv1, names=col_names)
    data2 = pandas.read_csv(csv2, names=col_names)
    data3 = pandas.read_csv(csv3, names=col_names)
    data4 = pandas.read_csv(csv4, names=col_names)

    df_list = [data0, data1, data2, data3, data4]
    final_df = pandas.DataFrame()

    # Extracts only MCC values and forms new DataFrame with said values.
    for df in df_list:
        if math.isnan(df['MCC'].iat[-1]):
            cat_stop = df['tn'].iat[-1]
            dog_units = df['fp'].iat[-1]
            df = df.drop(columns=['tn', 'fp', 'fn', 'tp', 'CK'])
            df = df[:-1]
            length = len(df)
            df.loc[length] = cat_stop
            df.loc[length + 1] = dog_units
        else:
            df = df.drop(columns=['tn', 'fp', 'fn', 'tp', 'CK'])

        final_df = pandas.concat([final_df, df], axis=1)

    # Saves new DataFrame to csv.
    if not os.path.isfile(final_csv):
        final_df.to_csv(final_csv, header=False, index=False)
    else:
        with open(final_csv, 'a') as f:
            final_df.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    generate_MCC_csv(csv0, csv1, csv2, csv3, csv4, final_csv)



