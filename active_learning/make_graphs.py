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
RESULT_PICKLE_DIR = 'uncert_5_times_28-03-2023-03:55:00'
RESULT_PICKLE_PATH = f'../result_pickles/{RESULT_PICKLE_DIR}'
GRAPH_SAVE_PATH = '../result' + f'/{RESULT_PICKLE_DIR}'
GRAPH_TITLE = 'Uncertainty Sampling\n Tested with UCI dataset'

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

  for i in range(result.shape[0]):
      avg = result[i][t]
      pool_result.append(avg)
  return pool_result

def output_graph(pool_results, init_size, metric):
  fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
  colors = ['red', 'blue', 'purple', 'orange', 'forestgreen', 'pink', 'peru', 'black']

  if 'committee' in RESULT_PICKLE_DIR:
    pattern = COMMITTEE_NUM_PATTERN
    label = 'VE '
  if 'uncert' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
    label = 'Pool '
  if 'batch' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
    label = 'Pool '

  for pool_result, size_pattern, color in zip(pool_results, pattern, colors):
      ax.plot(pool_result, label=label+str(size_pattern), color=color)
      ax.legend(loc=4)

  ax.set_ylim(bottom=0.50, top=1)
  ax.grid(True)

  title = f'{GRAPH_TITLE}: Init seed - {init_size}\n {metric}'
  ax.set_title(title)
  ax.set_xlabel('Query iteration')
  ax.set_ylabel(f'{metric} Score')

  save_dir = f'{GRAPH_SAVE_PATH}/{metric}'
  makedirs(save_dir, exist_ok=True)
  save_path = save_dir+f'/init_{init_size}.jpeg'
  plt.savefig(save_path, bbox_inches = 'tight')
  plt.close()
  #plt.show()

def make_graph(metric):
  for init_size in INIT_SEED_PATTERN:
      result = []

      pattern = None
      if 'committee' in RESULT_PICKLE_DIR:
        pattern = COMMITTEE_NUM_PATTERN
      if 'uncert' in RESULT_PICKLE_DIR:
        pattern = POOL_SIZE_PATTERN
      if 'batch' in RESULT_PICKLE_DIR:
        pattern = POOL_SIZE_PATTERN

      for size_pattern in pattern:
          if 'committee' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/*_{size_pattern}_cnum*'
          if 'uncert' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/*_{size_pattern}_pool*'
          if 'batch' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/{size_pattern}_pool/4_batch*'
          pickle_file_name = glob.glob(target)[0]
          print(f'Loading {pickle_file_name} ...')
          with open(pickle_file_name, 'rb') as f:
              rs = pickle.load(f)
              tmp = getDataByMetrc(rs, metric=metric.lower())
              result.append(tmp)
      output_graph(result, init_size, metric)
      print('Next Init ...')

def main():
  print(f'Make graphs for {RESULT_PICKLE_DIR} ...')
  for metric in ['Precision', 'Recall', 'F1', 'Accuracy']:
    make_graph(metric)  
  print('Finished')

if __name__ == '__main__':
  main()