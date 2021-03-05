from __future__ import print_function, division
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
try:
    import cPickle as pickle
except:
    import pickle
from sklearn.metrics import confusion_matrix
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import pandas as pd
import contextlib
import active_learners.helper as helper

def measure_exec_time(func):
    time_s = time.time()
    response = func()
    time_e = time.time() - time_s
    print('Exec time: ', time_e)
    return response


class ActiveLearner(object):
    def __init__(self):
        pass

    def query(self, context):
        pass

    def sample(self, context, N):
        pass

    def retrain(self, x, y):
        pass


class RandomSampler(ActiveLearner):
    def __init__(self, func):
        self.func = func
        self.name = 'random'
        self.total_good_samples = 0

    def query(self, context):
        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        x_star = np.random.uniform(xmin, xmax)
        self.total_good_samples += 1
        return np.hstack((x_star, context))

    def sample(self, context, N=1):
        return self.query(context)

    def reset_sample(self):
        self.total_good_samples = 0

    def retrain(self, newx=None, newy=None):
        pass


def run_ActiveLearner(active_learner, context, save_fnm, iters, random_context=False):
    '''
    Actively query a function with active learner.
    Args:
        active_learner: an ActiveLearner object.
        context: the current context we are testing for the function.
        save_fnm: a file name string to save the queries.
        iters: total number of queries.
    '''
    time_report_fnm = save_fnm+ '_time.txt'
    with contextlib.suppress(FileNotFoundError):
        os.remove(time_report_fnm)

    # Retrieve the function associated with active_learner
    func = active_learner.func
    # Queried x and y
    xq, yq = None, None
    # All the queries x and y
    xx = np.zeros((0, func.x_range.shape[1]))
    yy = np.zeros(0)
    # Start active queries
    for i in range(iters):

        if random_context:
            context = helper.gen_context(func)

        time_s = time.time()
        active_learner.retrain(xq, yq)
        time_r = time.time() - time_s

        time_s = time.time()
        xq = active_learner.query(context)
        time_q = time.time() - time_s

        yq = func(xq)
        xx = np.vstack((xx, xq))
        yy = np.hstack((yy, yq))
        print('i={}, xq={}, yq={}  [tr={}, tq={}]'.format(i, xq, yq, time_r, time_q))

        pickle.dump((xx, yy, context), open(save_fnm, 'wb'))
        pd.DataFrame([[time_r, time_q]]).to_csv(time_report_fnm, mode='a', index = False, header=False)
