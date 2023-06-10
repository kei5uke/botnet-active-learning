from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier as rf
import numpy as np
from datetime import datetime
import global_variables as g
from utils import file_management, split_seeds, print_class_num, get_train_test
import sys
import os
sys.path.append('../active_learning')
sys.path.append('../')


# Global Vars
QUERY_NUM = g.QUERY_NUM
REPEAT = g.REPEAT
INIT_SEED_PATTERN = g.INIT_SEED_PATTERN
POOL_SIZE_PATTERN = g.POOL_SIZE_PATTERN
CLASS_NUM = g.CLASS_NUM

# Shared Vars
TEST_RUN = False
TIME = datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
DF_FILENAME = '../../df_pickles/N-BaIoT_90000_23.pkl'
MULTICLASS = True
RESULT_DEFAULT_PATH = '../../result_pickles/'

if TEST_RUN:
    QUERY_NUM = 5
    REPEAT = 2
    INIT_SEED_PATTERN = [4, 8]
    POOL_SIZE_PATTERN = [12, 16, 20]
    RESULT_DEFAULT_PATH = '../../result_pickles/test_run/'

RESULT_DIR_NAME = f'random_{REPEAT}_times_{TIME}/'
RESULT_DIR = RESULT_DEFAULT_PATH + RESULT_DIR_NAME


def random_learn(query_num, X_init, y_init, X_pool, y_pool, X_test, y_test):
    print('===== Active Learning =====')

    clf = rf()
    clf = clf.fit(X=X_init, y=y_init)

    y_pred = clf.predict(X_test)
    score = np.append(
        accuracy_score(
            y_test, y_pred), precision_recall_fscore_support(
            y_test, y_pred, average='weighted')[
                :3])
    history = [score]

    random_indics = []
    for index in range(query_num):
        random_indics.append(np.random.randint(low=0, high=X_pool.shape[0]))
        X = np.append(X_init, X_pool[random_indics], axis=0)
        y = np.append(y_init, y_pool[random_indics], axis=0)

        cf = rf()
        cf = cf.fit(X=X, y=y)

        y_pred = cf.predict(X_test)
        score = np.append(
            accuracy_score(
                y_test, y_pred), precision_recall_fscore_support(
                y_test, y_pred, average='weighted')[
                :3])

        history.append(score)
        print(f'Query:{index+1} ...')
    return history
    print('==========================\n')


def main():
    print(f'Start Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
    X_train, X_test, y_train, y_test = get_train_test(DF_FILENAME, MULTICLASS)

    for init_size in INIT_SEED_PATTERN:
        for pool_size in POOL_SIZE_PATTERN:
            history = np.zeros((QUERY_NUM + 1, 4))
            for r in range(REPEAT):
                X_init, y_init, X_pool, y_pool = split_seeds(
                    CLASS_NUM, init_size, pool_size, X_train, y_train)
                result = random_learn(
                    QUERY_NUM,
                    X_init,
                    y_init,
                    X_pool,
                    y_pool,
                    X_test,
                    y_test)
                history = history + result
            history = history / REPEAT

            # Save result for every size pattern
            file_management(
                RESULT_DIR +
                f'{init_size}_init/',
                history,
                pool_size,
                'pool')

    print("=== FINISHED ===")
    print(f'Finished Script: {datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}')
    print(f'Output pickle path: {RESULT_DIR}')


if __name__ == "__main__":
    main()
