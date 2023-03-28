import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os import makedirs

def print_class_num(data):
  for index, class_num in enumerate(np.unique(data, return_counts=True)[1]):
    print(f'Label {index} has: {class_num}')

def get_init_stratified_indices(n_size, class_num, n_labeled_examples, y_train):
  training_indices = []
  equal_num = n_size//class_num
  for i in range(class_num):
    for j in range(equal_num):
      while True:
        training_indice = np.random.randint(low=0, high=n_labeled_examples)
        if y_train[training_indice] == i:
          training_indices.append(training_indice)
          break
  if len(training_indices) == 0: raise Exception("Something is wrong")
  return training_indices

def split_seeds(class_num, init_size, pool_size, X_train, y_train):
  n_labeled_examples = X_train.shape[0]

  # Pick Init Seed
  training_indices = get_init_stratified_indices(init_size, class_num, n_labeled_examples, y_train)
  X_init = X_train[training_indices]
  y_init = y_train[training_indices]

  # Delete the init from Train and store in Pool
  X_pool = np.delete(X_train, training_indices, axis=0)
  y_pool = np.delete(y_train, training_indices, axis=0)

  # Pick Pool 
  current_pool_size = X_pool.shape[0]
  pool_indices = get_init_stratified_indices(pool_size, class_num, current_pool_size, y_pool)
  X_pool = X_pool[pool_indices]
  y_pool = y_pool[pool_indices]

  print('===== Init & Pool Seed =====')
  print(f'X init shape:{X_init.shape}')
  print(f'y init shape:{y_init.shape}')
  print_class_num(y_init)
  print(f'\nX pool shape:{X_pool.shape}')
  print(f'y pool shape:{y_pool.shape}')
  print_class_num(y_pool)
  print('============================\n')

  return X_init, y_init, X_pool, y_pool

def file_management(main_path, history, size, size_name):
  history = np.array(history)
  print('===== History Shape =====')
  print(f'History shape:{history.shape}')
  print('=========================\n')

  # Make directory
  makedirs(main_path, exist_ok=True)

  # Pickle name
  file_name = f'{size}_{size_name}.pkl'
  path = main_path + file_name
  with open(path, 'wb') as f:
    pickle.dump(history, f)

def get_train_test(path, multiclass=True, uci_df_path=''):
  df = pd.read_pickle(path)
  X = df.iloc[:,:df.shape[1]-1]
  y = df.iloc[:,df.shape[1]-1]

  if multiclass == False:
    y = y.replace([2,3],1)

  X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(),
                                                    y.to_numpy(),
                                                    test_size=40000,
                                                    stratify=y)

  if uci_df_path != '':
    X_test, y_test = None, None
    df = pd.read_pickle(uci_df_path)
    df = stratify(df, [20000, 20000, 20000])
    X_test = df.iloc[:,:df.shape[1]-1].to_numpy()
    y_test = df.iloc[:,df.shape[1]-1].to_numpy()
                                        
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)                                      

  print('===== Train & Test Data =====')
  print(f'X train shape:{X_train.shape}')
  print(f'y train shape:{y_train.shape}')
  print_class_num(y_train)
  print(f'\nX test shape:{X_test.shape}')
  print(f'y test shape:{y_test.shape}')
  print_class_num(y_test)
  print('=============================\n')

  return X_train, X_test, y_train, y_test

def stratify(df, counts):
  labels = sorted(df['label'].unique())
  li = []
  for label, count in zip(labels, counts):
    tmp = df[df['label'] == label]
    tmp = tmp[:count]
    li.append(tmp)
  frame = pd.concat(li, axis=0, ignore_index=True)
  return frame