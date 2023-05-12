import sys, os
sys.path.append('../active_learning')
sys.path.append('../')
from active_learning_with_modAL import active_learning, make_learner
from utils import file_management, split_seeds, print_class_num, get_train_test
import global_variables as g

import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier as rf
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

# Global Vars
QUERY_NUM = g.QUERY_NUM
REPEAT = g.REPEAT
INIT_SEED_PATTERN = g.INIT_SEED_PATTERN
POOL_SIZE_PATTERN = g.POOL_SIZE_PATTERN
COMMITTEE_NUM_PATTERN = g.COMMITTEE_NUM_PATTERN
CLASS_NUM = g.CLASS_NUM

# Shared Vars
TEST_RUN = False
SAMPLE_METHOD = 'uncert'
TIME = datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
DF_FILENAME = '../../df_pickles/df_120000_21.pkl'
MULTICLASS = True
CLASS_NUM = 4
RESULT_DEFAULT_PATH  = '../../result_pickles/'
UCI_DF_PATH = '../../df_pickles/UCI_df_60030_21.pkl'

if TEST_RUN == True:
  QUERY_NUM = 5
  REPEAT = 2
  INIT_SEED_PATTERN = [4, 8]
  POOL_SIZE_PATTERN = [12, 16, 20]
  COMMITTEE_NUM_PATTERN = [2, 3]
  RESULT_DEFAULT_PATH  = '../../result_pickles/test_run/'

RESULT_DIR_NAME = f'{SAMPLE_METHOD}_{REPEAT}_times_{TIME}/'
RESULT_DIR =  RESULT_DEFAULT_PATH + RESULT_DIR_NAME

def main():
  print(f'Start Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
  X_train, X_test, y_train, y_test = get_train_test(path=DF_FILENAME, multiclass=MULTICLASS, uci_df_path=UCI_DF_PATH)
  
  for init_size in INIT_SEED_PATTERN:
    for pool_size in POOL_SIZE_PATTERN:
      history = np.zeros((QUERY_NUM+1, 4))
      for r in range(REPEAT):
        X_init, y_init, X_pool, y_pool = split_seeds(CLASS_NUM, init_size, pool_size, X_train, y_train)
        learner = make_learner(SAMPLE_METHOD, uncertainty_sampling, rf(), X_init, y_init)
        result = active_learning(learner, QUERY_NUM, X_pool, y_pool, X_test, y_test)
        history = history + result
      history = history / REPEAT
      
      # Save result for every size pattern
      file_management(RESULT_DIR+ f'{init_size}_init/' , history, pool_size, 'pool')

  print("=== FINISHED ===")
  print(f'Finished Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
  print(f'Output pickle path: {RESULT_DIR}')

if __name__ == "__main__":
  main()