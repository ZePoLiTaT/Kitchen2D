# Author: Zi Wang
try:
    import cPickle as pickle
except:
    import pickle
import os
import active_learners.helper as helper
from active_learners.active_learner import run_ActiveLearner


def measure_exec_time(func):
    time_s = time.time()
    response = func()
    time_e = time.time() - time_s
    print('Exec time: ', time_e)
    return response

def gen_data(expid, exp, n_data, save_fnm):
    '''
    Generate initial data for a function associated the experiment.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        n_data: number of data points to generate.
        save_fnm: a file name string where the initial data will be
        saved.
    '''
    print('Generating data...')    
    func = helper.get_func_from_exp(exp)
    xx, yy = helper.gen_data(func, n_data, parallel=True)
    pickle.dump((xx, yy), open(save_fnm, 'wb'))

def run_exp(expid, exp, method, n_init_data, iters):
    '''
    Run the active learning experiment.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        method: learning method, including 
            'nn_classification': a classification neural network 
                based learning algorithm that queries the input that has 
                the largest output.
            'nn_regression': a regression neural network based 
                learning algorithm that queries the input that has 
                the largest output.
            'gp_best_prob': a Gaussian process based learning algorithm
                that queries the input that has the highest probability of 
                having a positive function value.
            'gp_lse': a Gaussian process based learning algorithm called
                straddle algorithm. See B. Bryan, R. C. Nichol, C. R. Genovese, 
                J. Schneider, C. J. Miller, and L. Wasserman, "Active learning for 
                identifying function threshold boundaries," in NIPS, 2006.
            'random': an algorithm that query uniformly random samples.
        n_data: number of data points to generate.
        save_fnm: a file name string where the initial data will be
        saved.
    '''
    dirnm = helper.BASE_PATH
    if not os.path.isdir(dirnm):
        os.mkdir(dirnm)
    init_fnm = os.path.join(
            dirnm, '{}_init_data_{}.pk'.format(exp, expid))
    gen_data(expid, exp, n_init_data, init_fnm)

    initx, inity = pickle.load(open(init_fnm, 'rb'))

    func = helper.get_func_from_exp(exp)

    active_learner = helper.get_learner_from_method(method, initx, inity, func)

    # file name for saving the learning results
    learn_fnm = os.path.join(
            dirnm, '{}_{}_{}.pk'.format(exp, method, expid))

    # get a context
    context = helper.gen_context(func)

    # start running the learner
    run_ActiveLearner(active_learner, context, learn_fnm, iters)    
    print('Finished running the learning experiment with context...', context)

def sample_exp(expid, exp, method):
    '''
    Sample from the learned model.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        method: see run_exp.
    '''
    func = helper.get_func_from_exp(exp)    
    xx, yy, context = helper.get_xx_yy(expid, method, exp=exp)
    active_learner = helper.get_learner_from_method(method, xx, yy, func)
    active_learner.retrain()
    # Enable gui
    func.do_gui = True
    while input('Continue? [y/n]') == 'y':
        print('sampling')
        x = active_learner.sample(context)
        print('x: ', x)
        func(x)

if __name__ == '__main__':
    exp = 'pour'
    method = 'gp_lse' #'nn_regression'# 'gp_lse' #'random'
    expid = 3000
    n_init_data = 300
    iters = 3000
    run_exp(expid, exp, method, n_init_data, iters)
    sample_exp(expid, exp, method)
