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

# Shared Vars
RESULT_PICKLE_DIR = 'uncert_5_times_06-03-2023-23:06:21'
RESULT_PICKLE_PATH = f'../result_pickles/{RESULT_PICKLE_DIR}'
GRAPH_SAVE_PATH = '../result' + f'/{RESULT_PICKLE_DIR}'
GRAPH_TITLE = 'Uncertainty Sampling'

def get_index_by_type(avg_type, metric):
  index = None
  if avg_type == 'macro':
      index = 0
  elif avg_type == 'weighted':
      index = 4
  if metric == 'precision':
      index += 0
  elif metric == 'recall':
      index += 1
  elif metric == 'f1':
      index += 2
  elif metric == 'accuracy':
      index = 8
  if index == None:
      raise Exception("Something is wrong")
  #print(index)
  return index

def getAvg(result, avg_type, metric):
  pool_result = []
  t = get_index_by_type(avg_type, metric)
  for i in range(QUERY_NUM + 1):
      avg = 0
      for p in range(REPEAT):
          avg += result[p][i][t]
      avg /= REPEAT
      pool_result.append(avg)
  return pool_result

def output_graph(pool_results, init_size, avg_type, metric):
  fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
  colors = ['red', 'blue', 'purple', 'orange', 'forestgreen', 'pink', 'peru', 'black']

  for pool_result, pool_size, color in zip(pool_results, POOL_SIZE_PATTERN, colors):
      ax.plot(pool_result, label=f'Pool {pool_size}', color=color)
      ax.legend(loc=4)

  ax.set_ylim(bottom=0.50, top=1)
  ax.grid(True)

  title = f'{GRAPH_TITLE}: Init seed - {init_size}\n {avg_type} {metric}'
  ax.set_title(title)
  ax.set_xlabel('Query iteration')
  ax.set_ylabel(f'{metric} Score')

  save_dir = f'{GRAPH_SAVE_PATH}/{avg_type}/{metric}'
  makedirs(save_dir, exist_ok=True)
  save_path = save_dir+f'/init_{init_size}.jpeg'
  plt.savefig(save_path, bbox_inches = 'tight')
  plt.close()
  #plt.show()

def make_graph(avg_type, metric):
  for init_size in INIT_SEED_PATTERN:
      result = []
      for pool_size in POOL_SIZE_PATTERN:
          target = RESULT_PICKLE_PATH + f'/{init_size}_init/*_{pool_size}_pool*'
          pickle_file_name = glob.glob(target)[0]
          print(f'Loading {pickle_file_name} ...')
          with open(pickle_file_name, 'rb') as f:
              rs = pickle.load(f)
              tmp = getAvg(rs, avg_type=avg_type.lower(), metric=metric.lower())
              result.append(tmp)
      output_graph(result, init_size, avg_type, metric)
      print('Next Init ...')

def main():
  print(f'Make graphs for {RESULT_PICKLE_DIR} ...')
  for avg_type in ['Macro', 'Weighted']:
    for metric in ['Precesion', 'Recall', 'F1', 'Accuracy']:
        make_graph(avg_type, metric)
  print('Finished')

if __name__ == '__main__':
  main()