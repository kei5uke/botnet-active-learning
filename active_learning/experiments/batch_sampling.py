from modAL.batch import uncertainty_batch_sampling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
import pandas as pd
from datetime import datetime
import global_variables as g
from utils import file_management, split_seeds, print_class_num, get_train_test
from active_learning_with_modAL import active_learning, make_learner
import sys
import os
sys.path.append('../active_learning')
sys.path.append('../')


# Global Vars
QUERY_NUM = g.QUERY_NUM
REPEAT = g.REPEAT
INIT_SEED_PATTERN = g.INIT_SEED_PATTERN
POOL_SIZE_PATTERN = g.POOL_SIZE_PATTERN
COMMITTEE_NUM_PATTERN = g.COMMITTEE_NUM_PATTERN
CLASS_NUM = g.CLASS_NUM
BATCH_PATTERN = g.BATCH_PATTERN

# Shared Vars
TEST_RUN = False
SAMPLE_METHOD = 'batch'
TIME = datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
DF_FILENAME = '../../df_pickles/N-BaIoT_90000_23.pkl'
MULTICLASS = True
CLASS_NUM = 3
RESULT_DEFAULT_PATH = '../../result_pickles/'

if TEST_RUN:
    QUERY_NUM = 16
    REPEAT = 2
    INIT_SEED_PATTERN = [4, 8]
    POOL_SIZE_PATTERN = [16, 32]
    COMMITTEE_NUM_PATTERN = [2, 3]
    BATCH_PATTERN = [4, 8]
    RESULT_DEFAULT_PATH = '../../result_pickles/test_run/'

RESULT_DIR_NAME = f'{SAMPLE_METHOD}_{REPEAT}_times_{TIME}/'
RESULT_DIR = RESULT_DEFAULT_PATH + RESULT_DIR_NAME


def main():
    print(f'Start Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
    X_train, X_test, y_train, y_test = get_train_test(DF_FILENAME, MULTICLASS)

    for init_size in INIT_SEED_PATTERN:
        for pool_size in POOL_SIZE_PATTERN:
            for batch_size in BATCH_PATTERN:
                preset_batch = partial(
                    uncertainty_batch_sampling,
                    n_instances=batch_size)
                q_num = QUERY_NUM // batch_size
                history = np.zeros((q_num + 1, 4))
                for r in range(REPEAT):
                    X_init, y_init, X_pool, y_pool = split_seeds(
                        CLASS_NUM, init_size, pool_size, X_train, y_train)
                    learner = make_learner(
                        SAMPLE_METHOD, preset_batch, rf(), X_init, y_init)
                    result = active_learning(
                        learner, q_num, X_pool, y_pool, X_test, y_test)
                    history = history + result
                history = history / REPEAT

                # Save result for every size pattern
                file_management(
                    RESULT_DIR +
                    f'{init_size}_init/{pool_size}_pool/',
                    history,
                    batch_size,
                    'batch')
    print(history)

    print("=== FINISHED ===")
    print(f'Finished Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
    print(f'Output pickle path: {RESULT_DIR}')


if __name__ == "__main__":
    main()
