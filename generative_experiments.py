import os
import sys
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
import torch as torch
import tensorflow as tf

from generative_data import load_data_set
from generative_models import FixedVarianceNormalVAE, NormalVAE, StudentVAE, VariationalVarianceVAE, precision_prior_params

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from experiment_vae import detlefsen_vae_baseline

# minimum DoF to produce well-defined variances
MIN_DOF = 3.0
assert MIN_DOF > 2

# dictionary of methods to test
METHODS = [
    # Fixed Variance VAE baselines
    {'name': 'Fixed-Var. VAE (1.0)', 'mdl': FixedVarianceNormalVAE,
     'kwargs': {'variance': 1.0}},
    {'name': 'Fixed-Var. VAE (0.001)', 'mdl': FixedVarianceNormalVAE,
     'kwargs': {'variance': 1e-3}},

    # VAE with single decoder network both w/ and w/o batch normalization
    {'name': 'VAE', 'mdl': NormalVAE,
     'kwargs': {'split_decoder': False, 'batch_norm': False}},
    {'name': 'VAE + BN', 'mdl': NormalVAE,
     'kwargs': {'split_decoder': False, 'batch_norm': True}},

    # VAE with split decoder networks both w/ and w/o batch normalization
    {'name': 'VAE-Split', 'mdl': NormalVAE,
     'kwargs': {'split_decoder': True, 'batch_norm': False}},
    {'name': 'VAE-Split + BN', 'mdl': NormalVAE,
     'kwargs': {'split_decoder': True, 'batch_norm': True}},

    # Detlefsen Baseline
    {'name': 'Detlefsen', 'kwargs': dict()},

    # Takahashi baselines
    {'name': 'MAP-VAE', 'mdl': NormalVAE,
     'kwargs': {'split_decoder': True,  'a': MIN_DOF, 'b': 1e-3 * (MIN_DOF - 1)}},
    {'name': 'Student-VAE', 'mdl': StudentVAE,
     'kwargs': {'min_dof': MIN_DOF}},

    # Our Methods
    {'name': 'V3AE-VAP', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'VAP'}},
    {'name': 'V3AE-Gamma', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'Standard', 'a': MIN_DOF, 'b': 1e-3 * (MIN_DOF - 1)}},
    {'name': 'V3AE-VAMP', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'VAMP'}},
    {'name': 'V3AE-VAMP*', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'VAMP*'}},
    {'name': 'V3AE-xVAMP', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'xVAMP'}},
    {'name': 'V3AE-xVAMP*', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'xVAMP*'}},
    {'name': 'V3AE-VBEM', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'VBEM'}},
    {'name': 'V3AE-VBEM*', 'mdl': VariationalVarianceVAE,
     'kwargs': {'min_dof': MIN_DOF, 'prior_type': 'VBEM*', 'k': 10}},
]

# latent dimension per data set
DIM_Z = {'mnist': 10, 'fashion_mnist': 25, 'svhn_cropped': 32, 'celeb_a': 32}
ARCHITECTURE = {'mnist': 'dense', 'fashion_mnist': 'dense', 'svhn_cropped': 'convolution', 'celeb_a': 'convolution'}
NUM_MC_SAMPLES = {'mnist': 20, 'fashion_mnist': 20, 'svhn_cropped': 5, 'celeb_a': 5}


def run_vae_experiments(method, dataset, num_trials, mode):

    # establish experiment directory
    experiment_dir = 'vae'
    os.makedirs(os.path.join('results', experiment_dir), exist_ok=True)

    # make sure results subdirectory exists
    os.makedirs(os.path.join('results', experiment_dir, dataset), exist_ok=True)

    # create full file names
    logger_file = os.path.join('results', experiment_dir, dataset, method['name'] + '_metrics.pkl').replace('*', 't')
    plotter_file = os.path.join('results', experiment_dir, dataset, method['name'] + '_plots.pkl').replace('*', 't')
    nan_file = os.path.join('results', experiment_dir, dataset, method['name'] + '_nan_log.txt').replace('*', 't')

    # load results if we are resuming
    if mode == 'resume' and os.path.exists(logger_file) and os.path.exists(plotter_file):
        logger = pd.read_pickle(logger_file)
        with open(plotter_file, 'rb') as f:
            plotter = pickle.load(f)
        t_start = max(logger.index)
        print('Resuming', dataset, method['name'], 'at trial {:d}'.format(t_start + 2))

    # otherwise, initialize the loggers
    else:
        logger = pd.DataFrame(columns=['Method', 'LL', 'Best Epoch',
                                       'Mean Bias', 'Mean RMSE',
                                       'Var Bias', 'Var RMSE',
                                       'Sample Bias', 'Sample RMSE'])
        plotter = {'x': None, 'training': [], 'reconstruction': []}
        if os.path.exists(nan_file):
            os.remove(nan_file)
        t_start = -1

    # common configurations
    if method['kwargs'].get('prior_type') in {'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}:
        batch_size = 125
        epochs = 500
        patience = 25
        clip_value = 5.0
    else:
        batch_size = 250
        epochs = 1000
        patience = 50
        clip_value = None

    # load data
    train_set, test_set, info = load_data_set(data_set_name=dataset, px_family='Normal', batch_size=batch_size)

    # loop over the trials
    for t in range(t_start + 1, num_trials):
        print('\n***** Trial {:d}/{:d}:'.format(t + 1, num_trials), method['name'], '*****')

        # skip batch normalization for convolution architectures
        if ARCHITECTURE[dataset] == 'convolution' and '+ BN' in method['name']:
            print('skipping batch normalization--not supported with convolution architecture')
            continue

        # set random number seeds
        np.random.seed(t)
        tf.random.set_seed(t)
        torch.manual_seed(t)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # get prior parameters for precision
        a, b, u = precision_prior_params(data=train_set,
                                         num_classes=info.features['label'].num_classes,
                                         pseudo_inputs_per_class=10)

        # get sub-set of test set for results plotting
        if plotter['x'] is None:
            plotter['x'] = precision_prior_params(data=test_set,
                                                  num_classes=info.features['label'].num_classes,
                                                  pseudo_inputs_per_class=10)[-1]

        # baselines with separate code bases
        if method['name'] == 'Detlefsen':

            # run detlefsen baseline
            x_train = np.concatenate([x['image'] for x in train_set.as_numpy_iterator()], axis=0)
            x_test = np.concatenate([x['image'] for x in test_set.as_numpy_iterator()], axis=0)
            hist = None
            metrics, reconstruction = detlefsen_vae_baseline(x_train=x_train, x_test=x_test, x_plot=plotter['x'],
                                                             dim_z=DIM_Z[dataset], epochs=epochs, batch_size=batch_size)
            metrics.update({'Method': method['name']})

        # otherwise run our methods
        else:

            # update kwargs accordingly
            kwargs = copy.deepcopy(method['kwargs'])
            kwargs.update({'dim_x': info.features['image'].shape, 'dim_z': DIM_Z[dataset],
                           'architecture': ARCHITECTURE[dataset], 'num_mc_samples': NUM_MC_SAMPLES[dataset], 'u': u})

            # configure and compile model
            mdl = method['mdl'](**kwargs)
            mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipvalue=clip_value), loss=[None])

            # train
            hist = mdl.fit(train_set, validation_data=test_set, epochs=epochs, verbose=1,
                           validation_steps=np.ceil(info.splits['test'].num_examples // batch_size),
                           callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                      tf.keras.callbacks.EarlyStopping(monitor='val_LPPL',
                                                                       min_delta=0.5,
                                                                       patience=patience,
                                                                       mode='max',
                                                                       restore_best_weights=True)])
            # print and log NaNs
            if sum(np.isnan(hist.history['loss'])):
                print('**** NaN Detected ****')
                print(dataset, method['name'], 'trial = {:d}'.format(t + 1), file=open(nan_file, 'a'))

            # retrieve best attained posterior predictive log likelihood on the validation data
            i_best = np.nanargmax(hist.history['val_LPPL'])
            elbo = hist.history['val_ELBO'][i_best]
            lppl = hist.history['val_LPPL'][i_best]

            # log scalar performance metrics
            num_pixels = 0
            mean_bias = 0
            mean_mse = 0
            var_bias = 0
            var_mse = 0
            sample_bias = 0
            sample_mse = 0
            for batch in test_set:
                x_mean, x_std, x_new = mdl.posterior_predictive_checks(batch['image'])
                num_pixels += np.prod(batch['image'].shape)
                mean_residuals = x_mean - batch['image']
                mean_bias += tf.reduce_sum(mean_residuals)
                mean_mse += tf.reduce_sum(mean_residuals ** 2)
                var_residuals = x_std ** 2 - mean_residuals ** 2
                var_bias += tf.reduce_sum(var_residuals)
                var_mse += tf.reduce_sum(var_residuals ** 2)
                sample_residuals = x_new - batch['image']
                sample_bias += tf.reduce_sum(sample_residuals)
                sample_mse += tf.reduce_sum(sample_residuals ** 2)
            mean_bias /= num_pixels
            mean_mse /= num_pixels
            var_bias /= num_pixels
            var_mse /= num_pixels
            sample_bias /= num_pixels
            sample_mse /= num_pixels

            # assemble metric and reconstruction dictionaries
            metrics = {'Method': method['name'], 'ELBO': elbo, 'LL': lppl, 'Best Epoch': i_best + 1,
                       'Mean Bias': mean_bias.numpy(), 'Mean RMSE': mean_mse.numpy() ** 0.5,
                       'Var Bias': var_bias.numpy(), 'Var RMSE': var_mse.numpy() ** 0.5,
                       'Sample Bias': sample_bias.numpy(), 'Sample RMSE': sample_mse.numpy() ** 0.5}
            x_mean, x_std, x_new = mdl.posterior_predictive_checks(x=plotter['x'])
            reconstruction = {'mean': x_mean, 'std': x_std, 'sample': x_new}

        # log/print scalar metrics
        new_df = pd.DataFrame(data=metrics, index=[t])
        logger = logger.append(new_df)
        print(new_df.to_string())

        # save training history and plot data
        if hist is not None:
            plotter['training'].append(hist.history)
        plotter['reconstruction'].append(reconstruction)

        # save results after each trial
        logger.to_pickle(logger_file)
        with open(plotter_file, 'wb') as f:
            pickle.dump(plotter, f)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='celeb_a', help='www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--num_trials', type=int, default=1, help='number of trials')
    parser.add_argument('--mode', type=str, default='resume', help='mode in {replace, resume}')
    parser.add_argument('--seed_init', default=1234, type=int, help='random seed init, multiplied by trial number')
    args = parser.parse_args()

    # check inputs
    assert args.dataset in DIM_Z.keys()
    assert args.mode in {'replace', 'resume'}

    # make result directory if it doesn't already exist
    os.makedirs('results', exist_ok=True)

    # run experiments accordingly
    for m in METHODS:
        run_vae_experiments(method=m, dataset=args.dataset, num_trials=args.num_trials, mode=args.mode)
