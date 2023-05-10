import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from modAL.models import ActiveLearner, Committee


def make_learner(sample_method, query_strat, clf, X_init, y_init, c_num=None):
  if sample_method == 'committee':
    learner_list = list()
    for member_idx in range(c_num):
      al = ActiveLearner(
          estimator=clf,
          X_training=X_init, y_training=y_init
      )
      learner_list.append(al)
    learner = Committee(learner_list=learner_list,
                        query_strategy=query_strat)

  elif sample_method == 'uncert' or sample_method == 'batch':
    learner = ActiveLearner(estimator=clf,
                            query_strategy=query_strat,
                            X_training=X_init, y_training=y_init)
  
  return learner

def active_learning(learner, query_num, X_pool, y_pool, X_test, y_test):
  print('===== Active Learning =====')

  y_pred = learner.predict(X_test)
  score = np.append(accuracy_score(y_test, y_pred),
                    precision_recall_fscore_support(y_test, y_pred, average='weighted')[:3])
  history = [score]
  for index in range(query_num):
    query_index, query_instance = learner.query(X_pool)

    X, y = X_pool[query_index], y_pool[query_index]
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    y_pred = learner.predict(X_test)
    score = np.append(accuracy_score(y_test, y_pred),
                      precision_recall_fscore_support(y_test, y_pred, average='weighted')[:3])
    history.append(score)
    print(f'Query:{index+1} ...')

  print('==========================\n')

  return np.array(history)

