import os
import sys
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_variables as g

CP_PATH = '../df_pickles/df_120000_21.pkl.f64'


def sqrt_col(col):
  if '_variance' in col.name:
    return np.sqrt(col)
  else:
    return col

def format_df(df):
  df = df.filter(regex='^(?!H_)') # H_ is not needed
  df.columns = df.columns.str.replace('L', '') # L part is deleted

  # apply the function to each column in the dataframe
  #df = df.apply(sqrt_col, axis=0)
  df.columns = df.columns.str.replace('_variance', '_std') # Now its std
  return df

def make_df(path_list, n_rows):
  li = []

  for filename in path_list:
    print(f'Loading {filename} ...')
    df = pd.read_csv(filename,
                      index_col=None,
                      header=0,
                      dtype=np.float64)
    df = df.sample(n_rows)
    df['device'] = filename.split('/')[3]
    df['traffic_type'] = filename.split('/')[-1].split('.')[0]
    if 'benign' in filename:
      df['label'] = 0
    elif 'gafgyt' in filename:
      df['label'] = 1
    elif 'mirai' in filename:
      df['label'] = 3

    li.append(df)

  frame = pd.concat(li, axis=0, ignore_index=True)
  return frame

def main():
  # Paths
  benign_paths = glob.glob("../dataset/UCI_Dataset/*/benign/*.csv")
  mirai_paths = glob.glob("../dataset/UCI_Dataset/*/mirai/*.csv")
  gafgyt_paths = glob.glob("../dataset/UCI_Dataset/*/gafgyt/*.csv")
  # Make DFs
  benign_df = make_df(benign_paths, 2500)
  mirai_df = make_df(mirai_paths, 667)
  gafgyt_df = make_df(gafgyt_paths, 572)
  # Concat
  df = pd.concat([benign_df, mirai_df, gafgyt_df], axis=0, ignore_index=True)
  df = format_df(df)
  
  # Extract the same columns from df
  cp_df = pd.read_pickle(CP_PATH)
  shared_columns = []
  count = 0
  for c in df.columns:
    for p in cp_df.columns:
      if c in p:
        count += 1
        shared_columns.append(c)
  if count != len(cp_df.columns):
    print('Warning: columns size does not match ...')

  df = df[shared_columns]

  print(f'SHAPE:{df.shape}')
  for label in df['label'].unique():
    print("LABEL " + str(label) + " SIZE:" + str(df[df['label'] == label].count()[1]))

  # Save as pickle
  save_name = f'../df_pickles/UCI_df_{df.shape[0]}_{df.shape[1]}_no_mod.pkl'
  if os.path.exists(save_name):
    print(f'File {save_name} already exists')
    save_name += '.copy' 
    df.to_pickle(save_name)
  else:
    df.to_pickle(save_name)
  print(f'Output path: {save_name}')
    
    
if __name__ == "__main__":
    main()