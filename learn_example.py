# Author: Zi Wang
try:
    import cPickle as pickle
except:
    import pickle
import os
import active_learners.helper as helper
from active_learners.active_learner import run_ActiveLearner
import time

def measure_exec_time(func):
    time_s = time.time()
    response = func()
    time_e = time.time() - time_s
    print('Exec time: ', time_e)
    return response

def gen_data(expid, exp, n_data, save_fnm, cpu_n=4):
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

    xx, yy = helper.gen_data(func, n_data, cpu_n=cpu_n)
    pickle.dump((xx, yy), open(save_fnm, 'wb'))

def run_exp(expid, exp, method, n_init_data, iters, context=None):
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
    random_context = context=='random'
    if not context or random_context:
        context = helper.gen_context(func)

    # start running the learner
    run_ActiveLearner(active_learner, context, learn_fnm, iters, random_context=random_context)    
    print('Finished running the learning experiment with context...', context)

def sample_exp(expid, exp, method, random_context=False):
    '''
    Sample from the learned model.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        method: see run_exp.
    '''
    func = helper.get_func_from_exp(exp)   
    xx, yy, context = helper.get_xx_yy(expid, method, exp=exp)
    print('Loaded context: ', context)
    active_learner = helper.get_learner_from_method(method, xx, yy, func)
    active_learner.retrain()

    # Enable gui
    func.do_gui = True
    while input('Continue? [y/n]') == 'y':

        if random_context:
            context = helper.gen_context(func)
            print('New context: ', context)

        active_learner.is_adaptive = True
        print('sampling adaptive? ', active_learner.is_adaptive)
        ts = time.time()
        x = active_learner.sample(context)
        te = time.time()-ts
        y = func(x)
        print('x: {}, y: {} [Time: {}]'.format(x, y, te))
        

if __name__ == '__main__':
    exp = 'pour'
    method = 'gp_lse' #'nn_regression'# 'gp_lse' #'random'
    # expid = '2k_fix_ctx_4'
    expid = '2k_fix_ctx_8'
    # expid = '10'
    # expid = '2k_rand_ctx'
    # expid = '2k_rand_ctx'

    n_init_data = 100 #100 #300
    iters = 1900

    # context = [4.833278658190212, 4.804885678143026, 3.9830618868497387, 3.2334355753690964]
    # context = [4.743738087004583,4.943091337192776,3.3374507988113242,3.010520430848918]
    # context = [4.5389033420550895, 4.232835173028836, 3.0997657305491124, 4.190558602827117]
    # context = [4.678802287108174,4.564399297202194,3.9451703815858084,3.9258270969416946]

    # context = [4.385101819756269,4.936988735590781,3.291328930704763,4.765813671565134]

    # context = [4.401222524477312,4.63263266270377,3.1421457242848048,3.354042045100015]
    context = [4.629142099063606,4.115335185306216,3.210574332496111,3.296484717039732]
    # context = [4.219517898660544,4.18672024385849,3.6066009654135516,3.2194459009840086]
    # context = 'random'

    run_exp(expid, exp, method, n_init_data, iters, context)

    sample_exp(expid, exp, method, random_context=False)
