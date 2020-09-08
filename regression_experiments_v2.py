import os
import sys
import torch
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

# import data loaders and callbacks
from regression_data import generate_toy_data
from callbacks import RegressionCallback

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from toy_regression import detlefsen_toy_baseline
from experiment_regression import detlefsen_uci_baseline

# import our regression models
from regression_models import prior_params, NormalRegressionWithVariationalPrecision

# results directory
RESULTS_DIR = 'resultsV2'


class MeanVarianceLogger(object):
    def __init__(self, df_data=None, df_eval=None):
        self.cols_data = ['Algorithm', 'Prior', 'x', 'y']
        self.df_data = pd.DataFrame(columns=['Algorithm', 'Prior', 'x', 'y']) if df_data is None else df_data
        self.cols_eval = ['Algorithm', 'Prior', 'x', 'mean(y|x)', 'std(y|x)']
        self.df_eval = pd.DataFrame(columns=self.cols_eval) if df_eval is None else df_eval

    @staticmethod
    def __to_list(val):
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        assert isinstance(val, np.ndarray)
        val = np.squeeze(val)
        return val.tolist()

    def update(self, algorithm, prior, x_train, y_train, x_eval, mean, std, trial):

        # update training points data frame
        algorithm_list = [algorithm] * len(x_train)
        prior_list = [prior] * len(x_train)
        x_train = self.__to_list(x_train)
        y_train = self.__to_list(y_train)
        df_new = pd.DataFrame(dict(zip(self.cols_data, (algorithm_list, prior_list, x_train, y_train))),
                              index=[trial] * len(x_train))
        self.df_data = self.df_data.append(df_new)

        # update evaluation data frame
        algorithm_list = [algorithm] * len(x_eval)
        prior_list = [prior] * len(x_eval)
        x_eval = self.__to_list(x_eval)
        mean = self.__to_list(mean)
        std = self.__to_list(std)
        df_new = pd.DataFrame(dict(zip(self.cols_eval, (algorithm_list, prior_list, x_eval, mean, std))),
                              index=[trial] * len(x_eval))
        self.df_eval = self.df_eval.append(df_new)


def train_and_eval(dataset, prior_type, prior_fam, epochs, batch_size, x_train, y_train, x_eval, y_eval, parallel, **kwargs):

    # toy data configuration
    if dataset == 'toy':

        # hyper-parameters
        d_hidden = 50
        f_hidden = 'sigmoid'
        learning_rate = 5e-3
        num_mc_samples = 50
        early_stopping = False

        # prior parameters
        u = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), 20), axis=-1)
        a, b = prior_params(kwargs.get('precisions'), prior_fam)

    # UCI data configuration
    else:

        # hyper-parameters
        d_hidden = 100 if dataset in {'protein', 'year'} else 50
        f_hidden = 'relu'
        learning_rate = 1e-4 if prior_type == 'MLE' else 1e-3
        num_mc_samples = 20
        early_stopping = True

        # prior parameters
        u = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
        a = kwargs.get('a')
        b = kwargs.get('b')

    # create TF data loaders
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train})
    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)
    ds_eval = tf.data.Dataset.from_tensor_slices({'x': x_eval, 'y': y_eval})
    ds_eval = ds_eval.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)

    # configure the model
    mdl = NormalRegressionWithVariationalPrecision(d_in=x_train.shape[1],
                                                   d_hidden=d_hidden,
                                                   f_hidden=f_hidden,
                                                   d_out=y_train.shape[1],
                                                   prior_type=prior_type,
                                                   prior_fam=prior_fam,
                                                   y_mean=0.0 if dataset == 'toy' else np.mean(y_train, axis=0),
                                                   y_var=1.0 if dataset == 'toy' else np.var(y_train, axis=0),
                                                   a=a,
                                                   b=b,
                                                   k=u.shape[0],
                                                   u=u,
                                                   n_mc=num_mc_samples)

    # train the model
    callbacks = [RegressionCallback(epochs, parallel)]
    if early_stopping:
        callbacks += [tf.keras.callbacks.EarlyStopping(monitor='val_LL', min_delta=1e-4, patience=500, mode='max')]
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5), loss=[None])
    hist = mdl.fit(ds_train, validation_data=ds_eval, epochs=epochs, verbose=0, callbacks=callbacks)

    # test for NaN's
    nan_detected = bool(np.sum(np.isnan(hist.history['loss'])))

    # get index of best validation log likelihood
    i_best = np.nanargmax(hist.history['val_LL'])
    if i_best != np.nanargmax(hist.history['val_LL (adjusted)']):
        delta = np.abs(np.max(hist.history['val_LL (adjusted)']) - hist.history['val_LL (adjusted)'][i_best])
        warnings.warn('LL != LL adjusted by {:f}'.format(delta))
    if np.nanargmax(hist.history['val_LL (adjusted)']) >= 0.9 * epochs:
        warnings.warn('LL not converged!')

    # retrieve performance metrics
    ll = hist.history['val_LL (adjusted)'][i_best]
    rmse = np.sqrt(hist.history['val_MSE'][i_best])

    # evaluate predictive mean and variance
    mdl.num_mc_samples = 2000
    mdl_mean, mdl_std = mdl.posterior_predictive_mean(x_eval).numpy(), mdl.posterior_predictive_std(x_eval).numpy()

    return ll, rmse, mdl_mean, mdl_std, nan_detected


def run_experiments(algorithm, dataset, batch_iterations, mode='resume', parallel=False, **kwargs):
    assert algorithm in {'Detlefsen', 'Detlefsen (fixed)', 'Gamma-Normal', 'LogNormal-Normal'}
    assert not (algorithm == 'Detlefsen (fixed)' and dataset != 'toy')
    assert mode in {'replace', 'resume'}

    # parse algorithm/prior names
    if algorithm == 'Gamma-Normal':
        prior_fam = 'Gamma'
        prior_type = kwargs.pop('prior_type')
        base_name = algorithm + '_' + prior_type
    elif algorithm == 'LogNormal-Normal':
        prior_fam = 'LogNormal'
        prior_type = kwargs.pop('prior_type')
        base_name = algorithm + '_' + prior_type
    else:
        prior_fam = ''
        prior_type = 'N/A'
        base_name = algorithm

    # parse prior type hyper-parameters
    if prior_type == 'Standard' and dataset != 'toy':
        hyper_params = '_' + kwargs.get('a') + '_' + kwargs.get('b')
    elif 'VAMP' in prior_type or 'VBEM' in prior_type:
        hyper_params = '_' + str(kwargs.get('k'))
    else:
        hyper_params = ''
    base_name += hyper_params
    base_name = base_name.replace(' ', '').replace('*', 't')

    # dataset specific hyper-parameters
    n_trials = 5 if dataset in {'protein', 'year'} else 20
    batch_size = 500 if dataset == 'toy' else 512

    # establish experiment directory
    experiment_dir = 'regression_toy' if dataset == 'toy' else 'regression_uci'
    experiment_dir += '_{:d}'.format(batch_iterations)
    os.makedirs(os.path.join(RESULTS_DIR, experiment_dir), exist_ok=True)

    # make sure results subdirectory exists
    os.makedirs(os.path.join(RESULTS_DIR, experiment_dir, dataset), exist_ok=True)

    # create full file names
    logger_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '.pkl')
    nan_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_nan_log.txt')
    data_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_data.pkl')
    mv_file = os.path.join(RESULTS_DIR, experiment_dir, dataset, base_name + '_mv.pkl')

    # load results if we are resuming
    if mode == 'resume' and os.path.exists(logger_file):
        logger = pd.read_pickle(logger_file)
        if dataset == 'toy':
            mv_logger = MeanVarianceLogger(df_data=pd.read_pickle(data_file), df_eval=pd.read_pickle(mv_file))
        t_start = max(logger.index)
        print('Resuming', dataset, prior_fam, prior_type, 'at trial {:d}'.format(t_start + 2))

    # otherwise, initialize the loggers
    else:
        logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'Hyper-Parameters', 'LL', 'RMSE'])
        if os.path.exists(nan_file):
            os.remove(nan_file)
        if dataset == 'toy':
            mv_logger = MeanVarianceLogger()
        t_start = -1

    # loop over the trials
    for t in range(t_start + 1, n_trials):
        if not parallel:
            print('\n*****', dataset, 'Trial {:d}/{:d}:'.format(t + 1, n_trials), algorithm, prior_type, '*****')

        # set random number seeds
        np.random.seed(t)
        tf.random.set_seed(t)
        torch.manual_seed(t)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # toy data
        if dataset == 'toy':

            # generate data
            x_train, y_train, x_eval, y_eval, true_std = generate_toy_data()

            # compute true precisions
            kwargs.update({'precisions': 1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2})

        # uci data
        else:

            # load and split data
            with open(os.path.join('data', dataset, dataset + '.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
            x, y = data_dict['data'], data_dict['target']
            x_train, x_eval, y_train, y_eval = skl.model_selection.train_test_split(x, y, test_size=0.1)

            # scale features
            x_scale = skl.preprocessing.StandardScaler().fit(x_train)
            x_train = x_scale.transform(x_train)
            x_eval = x_scale.transform(x_eval)

        # compute epochs to correspond to the number of batch iterations (as used by Detlefsen)
        epochs = round(batch_iterations / int(np.ceil(x_train.shape[0] / batch_size)))

        # run appropriate algorithm
        if algorithm == 'Detlefsen' and dataset == 'toy':
            ll, rmse, mean, std = detlefsen_toy_baseline(x_train, y_train, x_eval, y_eval, bug_fix=False)

        elif algorithm == 'Detlefsen (fixed)' and dataset == 'toy':
            ll, rmse, mean, std = detlefsen_toy_baseline(x_train, y_train, x_eval, y_eval, bug_fix=True)

        elif algorithm == 'Detlefsen' and dataset != 'toy':
            ll, rmse = detlefsen_uci_baseline(x_train, y_train, x_eval, y_eval, batch_iterations, batch_size)

        else:
            ll, rmse, mean, std, nans = train_and_eval(dataset, prior_type, prior_fam, epochs, batch_size, x_train, y_train, x_eval, y_eval, parallel, **kwargs)
            print(dataset, algorithm, prior_type, '{:d}/{:d}:'.format(t + 1, n_trials), 'LL Exact:', ll, 'RMSE:', rmse)
            if nans:
                print('**** NaN Detected ****')
                print(dataset, prior_fam, prior_type, t + 1, file=open(nan_file, 'a'))

        # save results
        new_df = pd.DataFrame({'Algorithm': algorithm, 'Prior': prior_type, 'LL': ll, 'RMSE': rmse}, index=[t])
        logger = logger.append(new_df)
        logger.to_pickle(logger_file)
        if dataset == 'toy':
            mv_logger.update(algorithm, prior_type, x_train, y_train, x_eval, mean, std, trial=t)
            mv_logger.df_data.to_pickle(data_file)
            mv_logger.df_eval.to_pickle(mv_file)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Detlefsen', help='algorithm')
    parser.add_argument('--dataset', type=str, default='boston', help='data set name = {toy} union UCI sets')
    parser.add_argument('--batch_iterations', type=int, default=int(20e4), help='batch iterations')
    parser.add_argument('--mode', type=str, default='resume', help='mode in {replace, resume}')
    parser.add_argument('--prior_type', type=str, help='prior type')
    parser.add_argument('--a', type=float, help='standard prior parameter')
    parser.add_argument('--b', type=float, help='standard prior parameter')
    parser.add_argument('--k', type=int, help='number of mixing prior components')
    parser.add_argument('--parallel', type=int, default=0, help='adjust console print out for parallel runs')
    args = parser.parse_args()

    # check inputs
    assert args.dataset in {'toy'}.union(set(os.listdir('data')))

    # assemble configuration dictionary
    KWARGS = {}
    if args.prior_type is not None:
        KWARGS.update({'prior_type': args.prior_type})
    if args.a is not None:
        KWARGS.update({'a': args.a})
    if args.b is not None:
        KWARGS.update({'b': args.b})
    if args.k is not None:
        KWARGS.update({'k': args.k})

    # make result directory if it doesn't already exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # run experiments
    run_experiments(args.algorithm, args.dataset, args.batch_iterations, args.mode, bool(args.parallel), **KWARGS)
