import os
import sys
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_variables as g

# Global Var
LABEL_RATIO = g.LABEL_RATIO

# Shared Var
FILE_INCLUDE_PERCENTAGE = 0.2
REMOVE_CORR_PER = 0.80


def make_df(path_list):
    li = []

    for filename in path_list:
        print(f'Loading {filename} ...')
        df = pd.read_csv(
            filename,
            index_col=None,
            header=0,
            dtype=np.float32,
            skiprows=lambda x: x > 0 and random.random() >= FILE_INCLUDE_PERCENTAGE)
        if 'normal' in filename:
            df['label'] = 0
        elif 'bashlite' in filename:
            df['label'] = 1
        elif 'torii' in filename:
            df['label'] = 2
        elif 'mirai' in filename:
            df['label'] = 3

        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame


def remove_corr(df, threshold):
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            break
        else:
            delete_column = None
            saved_column = None

            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column

            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)

    return df_corr.columns


def stratify(df, counts):
    labels = sorted(df['label'].unique())
    li = []
    for label, count in zip(labels, counts):
        tmp = df[df['label'] == label]
        tmp = tmp.sample(count)
        li.append(tmp)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def main():
    malware_paths = glob.glob("../dataset/MedBIoT_Dataset/malware_bulk/*.csv")
    benign_paths = glob.glob("../dataset/MedBIoT_Dataset/normal_bulk/*.csv")

    # Make a whole df
    df = make_df(malware_paths + benign_paths)
    # Stratify the size accordingly
    df = stratify(df, LABEL_RATIO)
    # Get columns with low-correlation
    left_columns = remove_corr(df[df.columns[:-1]], REMOVE_CORR_PER)
    columns = np.append(left_columns, 'label')
    df = df[columns]

    print(f'SHAPE:{df.shape}')
    for label in df['label'].unique():
        print("LABEL " + str(label) + " SIZE:" +
              str(df[df['label'] == label].count()[1]))

    # Save as pickle
    save_name = f'../df_pickles/df_{df.shape[0]}_{df.shape[1]}.pkl'
    if os.path.exists(save_name):
        print(f'File {save_name} already exists')
        save_name += '.copy'
        df.to_pickle(save_name)
    else:
        df.to_pickle(save_name)
    print(f'Output path: {save_name}')


if __name__ == "__main__":
    main()
