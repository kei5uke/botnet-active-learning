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
# US
# CU
# RESULT_PICKLE_DIR = 'uncert_5_times_15-03-2023-05:41:23'
# GRAPH_TITLE = 'Ucertainty Sampling: Classification Uncertainty'
# CM
# RESULT_PICKLE_DIR = 'uncert_5_times_22-04-2023-20:47:09'
# GRAPH_TITLE = 'N-BIoT\n Ucertainty Sampling: Classification Margin'
# CE
# RESULT_PICKLE_DIR = 'uncert_5_times_23-03-2023-19:13:36'
# GRAPH_TITLE = 'Ucertainty Sampling: Classification Entropy'

# QbC
# VE
# RESULT_PICKLE_DIR = 'committee_5_times_23-04-2023-22:17:30'
# GRAPH_TITLE = 'N-BIoT\n Query by Committee: Vote Entropy'
# CE
# RESULT_PICKLE_DIR = 'committee_5_times_21-03-2023-12:17:11'
# GRAPH_TITLE = 'Query by Committee: Consensus Entropy'
# MD
# RESULT_PICKLE_DIR = 'committee_5_times_25-03-2023-11:01:35'
# GRAPH_TITLE = 'Query by Committee: Max Disagreement'

# Ranked
# VE
# RESULT_PICKLE_DIR = 'batch_5_times_25-04-2023-11:06:00'
# SKIP = 4
# GRAPH_TITLE = f'Ranked Batch: {SKIP} Instances'

# Random
RESULT_PICKLE_DIR = 'random_5_times_22-04-2023-20:46:23'
GRAPH_TITLE = 'N-BIoT Random Sampling'


RESULT_PICKLE_PATH = f'../result_pickles/{RESULT_PICKLE_DIR}'
GRAPH_SAVE_PATH = '../result' + f'/{RESULT_PICKLE_DIR}'

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

  # delete after thesis
  if len(result.shape) == 3:
      for i in range(1000 + 1):
          avg = 0
          for p in range(5):
              avg += result[p][i][t]
          avg /= 5
          pool_result.append(avg)
  else:
      for i in range(result.shape[0]):
          avg = result[i][t]
          pool_result.append(avg)

  return pool_result

def get_pool_pattern_based_on_dir():
  if 'committee' in RESULT_PICKLE_DIR:
    pattern = COMMITTEE_NUM_PATTERN
    label='Committee '
  if 'uncert' in RESULT_PICKLE_DIR or 'random' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
    label='Pool '
  if 'batch' in RESULT_PICKLE_DIR:
    pattern = POOL_SIZE_PATTERN
    label='Pool '
  return pattern, label

def output_graph(pool_results, init_size, metric):
  fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
  colors = ['red', 'blue', 'purple', 'orange', 'forestgreen', 'peru', 'black', 'pink']

  pattern, label = get_pool_pattern_based_on_dir()
  # For random comparison
  for i, (pool_result, size_pattern, color) in enumerate(zip(pool_results, pattern+[8000], colors)):
  # for i, (pool_result, size_pattern, color) in enumerate(zip(pool_results, pattern, colors)):
      if i == len(pool_results) - 1:
        label= 'Random Sampling Pool '
        color='magenta'
      ax.plot(pool_result[:100], label=label+str(size_pattern), color=color, alpha=0.7)
      ax.legend(loc=4)

  ax.set_ylim(bottom=0.4, top=1.0)
  ax.grid(axis="x")
  ax.minorticks_on()
  ax.grid(which = "both", axis="y")

  title = f'{GRAPH_TITLE}\n Init seed: {init_size}'
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
      pattern, _ = get_pool_pattern_based_on_dir()
      for size_pattern in pattern:
          if 'committee' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/*{size_pattern}_cnum*'
          if 'uncert' in RESULT_PICKLE_DIR or 'random' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/{size_pattern}_pool*'
          if 'batch' in RESULT_PICKLE_DIR:
            target = RESULT_PICKLE_PATH + f'/{init_size}_init/{size_pattern}_pool/{SKIP}_batch*'
          print(target)

          pickle_file_name = glob.glob(target)[0]
          print(f'Loading {pickle_file_name} ...')
          with open(pickle_file_name, 'rb') as f:
              rs = pickle.load(f)
              tmp = getDataByMetrc(rs, metric=metric.lower())
              result.append(tmp)
      
      # For Comparering Random
      target = '../result_pickles/random_5_times_22-04-2023-20:46:23' + f'/{init_size}_init/8000_pool*'
      print(target)
      file_name = glob.glob(target)[0]
      print(f'Loading {file_name} ...')
      with open(file_name, 'rb') as f:
          rs = pickle.load(f)
          random_result = getDataByMetrc(rs, metric=metric.lower()) 

      # When comparering Ranked Batch
      # averages = []
      # skip = SKIP
      # for i in range(0, len(random_result), skip):
      #     average = sum(random_result[i:i+skip]) / len(random_result[i:i+skip])
      #     averages.append(average)
      # result.append(averages)

      output_graph(result, init_size, metric)
      print('Next Init ...')

def main():
  print(f'Make graphs for {RESULT_PICKLE_DIR} ...')
  for metric in ['Precision', 'Recall', 'F1', 'Accuracy']:
    make_graph(metric)  
  print('Finished')

if __name__ == '__main__':
  main()