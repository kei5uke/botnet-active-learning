import global_variables as g
import random
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from os import makedirs

# Global Vars
QUERY_NUM = g.QUERY_NUM
REPEAT = g.REPEAT
INIT_SEED_PATTERN = g.INIT_SEED_PATTERN
POOL_SIZE_PATTERN = g.POOL_SIZE_PATTERN
COMMITTEE_NUM_PATTERN = g.COMMITTEE_NUM_PATTERN
BATCH_PATTERN = g.BATCH_PATTERN

# Shared Vars
RESULT_PICKLE_DIR = 'batch_5_times_25-04-2023-11:06:00'
RESULT_PICKLE_PATH = f'../result_pickles/{RESULT_PICKLE_DIR}'
TMO=4

def get_index_by_type(metric):
  index = None
  if metric == 'precision':
      index = 1
  elif metric == 'recall':
      index = 2
  elif metric == 'f1':
      index = 3
  elif metric == 'accuracy':
      index = 0
  if index == None:
      raise Exception("Something is wrong")
  #print(index)
  return index

def getDataByMetrc(result, metric):
  pool_result = []
  t = get_index_by_type(metric)

  # for i in range(QUERY_NUM + 1):
  #     avg = 0
  #     for p in range(REPEAT):
  #         avg += result[p][i][t]
  #     avg /= REPEAT
  #     pool_result.append(avg)

  for i in range(result.shape[0]):
      avg = result[i][t]
      pool_result.append(avg)
  return pool_result

def get_info(pool_results, pattern, init_size):
  pool_results = np.array(pool_results)

  # Init comparison
  # max_value = np.max(pool_results)
  # max_index = np.unravel_index(np.argmax(pool_results), pool_results.shape)
  # print(f'Init {init_size}: Max Value: {max_value} Pool: {pattern[max_index[0]]} Query: {max_index[1]}')

  # # Init 0.9 Comparison
  # if init_size == 9:
  # indices = np.where(pool_results >= 0.9)
  # if indices[0].size > 0:
  #   max_index = (indices[0][0], indices[1][0])
  #   max_value = pool_results[max_index]
  #   print(f'Init {init_size}: Max Value: {max_value} Pool: {pattern[max_index[0]]} Query: {max_index[1]}')
  # else:
  #   print("No value over 0.9 found.")

  # Init 12 Pool Comparison
  if init_size == 9:
    pool_results = np.array(pool_results)
    for r, p in zip(pool_results, pattern):
      max_value = np.max(r)
      max_index = np.unravel_index(np.argmax(r), r.shape)
      print(f'Init {init_size}: Max Value: {max_value} Pool: {p} Query: {max_index[0]}')

  # pool_results = np.array(pool_results
  # if init_size == 12:
  #   for r, p in zip(pool_results, pattern):
  #     indices = np.where(r >= 0.99)
  #     # print(indices[0])
  #     if indices[0].size > 0:
  #       max_index = indices[0][0]
  #       max_value = r[max_index]
  #       print(f'Init {init_size}: Max Value: {max_value} Pool: {p} Query: {max_index}')

def output_graph(pool_results, init_size, metric):
  pattern = get_pool_pattern_based_on_dir()
  get_info(pool_results, pattern, init_size)

def get_pool_pattern_based_on_dir():
  if 'committee' in RESULT_PICKLE_DIR:
        pattern = COMMITTEE_NUM_PATTERN
  if 'uncert' in RESULT_PICKLE_DIR or 'random' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
  if 'batch' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
  return pattern

def make_graph(metric):
  for init_size in INIT_SEED_PATTERN:
      result = []

      pattern = get_pool_pattern_based_on_dir()

      for size_pattern in pattern:
          if 'committee' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/*{size_pattern}_cnum*'
          if 'uncert' in RESULT_PICKLE_DIR or 'random' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/{size_pattern}_pool*'
          if 'batch' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/{size_pattern}_pool/4_batch*'
          pickle_file_name = glob.glob(target)[0]
          with open(pickle_file_name, 'rb') as f:
              rs = pickle.load(f)
              tmp = getDataByMetrc(rs, metric=metric.lower())
              result.append(tmp)

      output_graph(result, init_size, metric)

def main():
  for metric in ['F1']:
    make_graph(metric)  

if __name__ == '__main__':
  main()