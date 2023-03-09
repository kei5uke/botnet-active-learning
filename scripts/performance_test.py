import global_variables as g
import random
import sys
import numpy as np
import pandas as pd
import pickle
from os import makedirs
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling

# Global Vars
QUERY_NUM = g.QUERY_NUM
REPEAT = g.REPEAT
INIT_SEED_PATTERN = g.INIT_SEED_PATTERN
POOL_SIZE_PATTERN = g.POOL_SIZE_PATTERN
COMMITTEE_NUM_PATTERN = g.COMMITTEE_NUM_PATTERN

# Shared Vars
TEST_RUN = True
SAMPLE_METHOD = 'uncert'
TIME = datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
DF_FILENAME = '../df_pickles/df_120000_20.pkl'
MULTICLASS = True
CLASS_NUM = 4
RESULT_DIR_NAME = f'{SAMPLE_METHOD}_{REPEAT}_times_{TIME}/'
RESULT_DIR = '../result_pickles/' + RESULT_DIR_NAME

if TEST_RUN == True:
  QUERY_NUM = 5
  REPEAT = 2
  INIT_SEED_PATTERN = [4, 8]
  POOL_SIZE_PATTERN = [12, 16, 20]
  COMMITTEE_NUM_PATTERN = [2, 3]
  RESULT_DIR_NAME = f'{SAMPLE_METHOD}_{REPEAT}_times_{TIME}/'
  RESULT_DIR = '../result_pickles/test_run/' + RESULT_DIR_NAME

def print_class_num(data):
  for index, class_num in enumerate(np.unique(data, return_counts=True)[1]):
    print(f'LABEL {index}: {class_num}')

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

def split_seeds(init_size, pool_size, X_train, y_train):
  n_labeled_examples = X_train.shape[0]

  # Pick Init Seed
  training_indices = get_init_stratified_indices(init_size, CLASS_NUM, n_labeled_examples, y_train)
  X_init = X_train[training_indices]
  y_init = y_train[training_indices]

  # Delete the init from Train and store in Pool
  X_pool = np.delete(X_train, training_indices, axis=0)
  y_pool = np.delete(y_train, training_indices, axis=0)

  # Pick Pool 
  current_pool_size = X_pool.shape[0]
  pool_indices = get_init_stratified_indices(pool_size, CLASS_NUM, current_pool_size, y_pool)
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

def make_learner(sample_method, clf, X_init, y_init, c_num=None):
  if SAMPLE_METHOD == 'committee':
    learner_list = list()
    for member_idx in range(c_num):
      learner = ActiveLearner(
          estimator=clf,
          X_training=X_init, y_training=y_init
      )
      learner_list.append(learner)
    learner = Committee(learner_list=learner_list)

  elif SAMPLE_METHOD == 'uncert':
    learner = ActiveLearner(estimator=clf, X_training=X_init, y_training=y_init)
  
  return learner

def active_learning(learner, X_init, y_init, X_pool, y_pool, X_test, y_test):
  print('===== Active Learning =====')

  y_pred = learner.predict(X_test)
  score = np.append(precision_recall_fscore_support(y_test, y_pred, average='macro'),
                    precision_recall_fscore_support(y_test, y_pred, average='weighted'))
  score = np.append(score, accuracy_score(y_test, y_pred))
  history = [score]
  for index in range(QUERY_NUM):
    query_index, query_instance = learner.query(X_pool)

    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    y_pred = learner.predict(X_test)
    score = np.append(precision_recall_fscore_support(y_test, y_pred, average='macro'),
                    precision_recall_fscore_support(y_test, y_pred, average='weighted'))
    score = np.append(score, accuracy_score(y_test, y_pred))
    history.append(score)
    print(f'Query:{index+1} ...')
  print('==========================\n')
    
  return history

def file_management(history, init_size, pool_size=None, c_num=None):
  history = np.array(history)
  print('===== History Shape =====')
  print(f'History shape:{history.shape}')
  print('=========================\n')

  # Dir for the init
  dir_name = RESULT_DIR + f'{init_size}_init/'
  makedirs(dir_name, exist_ok=True)

  # Pickle name
  if pool_size != None:
    file_name = f'{init_size}_init_{pool_size}_pool.pkl'
  elif c_num != None:
    file_name = f'{init_size}_init_{c_num}_cnum.pkl'
  path = dir_name + file_name

  with open(path, 'wb') as f:
    pickle.dump(history, f)

def main():
  print(f'Start Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
  df = pd.read_pickle(DF_FILENAME)
  X = df.iloc[:,:df.shape[1]-2]
  y = df.iloc[:,df.shape[1]-1]

  if MULTICLASS == False:
    y = y.replace([2,3],1)

  X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(),
                                                    y.to_numpy(),
                                                    test_size=40000,
                                                    stratify=y)

  print('===== Train & Test Data =====')
  print(f'X train shape:{X_train.shape}')
  print(f'y train shape:{y_train.shape}')
  print_class_num(y_train)
  print(f'\nX test shape:{X_test.shape}')
  print(f'y test shape:{y_test.shape}')
  print_class_num(y_test)
  print('=============================\n')

  if SAMPLE_METHOD == 'uncert':
    for init_size in INIT_SEED_PATTERN:
      for pool_size in POOL_SIZE_PATTERN:
        history = []
        for r in range(REPEAT):
          X_init, y_init, X_pool, y_pool = split_seeds(init_size, pool_size, X_train, y_train)
          learner = make_learner(SAMPLE_METHOD, rf(), X_init, y_init)
          result = active_learning(learner, X_init, y_init, X_pool, y_pool, X_test, y_test)
          history.append(result)
    
        # Save result for every size pattern
        file_management(history, init_size, pool_size)

  if SAMPLE_METHOD == 'committee':
    for init_size in INIT_SEED_PATTERN:
      pool_size = 7000
      for c_num in COMMITTEE_NUM_PATTERN:
        history = []
        for r in range(REPEAT):
          X_init, y_init, X_pool, y_pool = split_seeds(init_size, pool_size, X_train, y_train)
          learner = make_learner(SAMPLE_METHOD, rf(), X_init, y_init, c_num)
          result = active_learning(learner, X_init, y_init, X_pool, y_pool, X_test, y_test)
          history.append(result)
    
        # Save result for every size pattern
        file_management(history, init_size, None, c_num)


  print("=== FINISHED ===")
  print(f'Finished Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
  print(f'Output pickle path: {RESULT_DIR}')



if __name__ == "__main__":
  main()
