import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind_from_stats


def raw_result_table(pickle_files, main_body):

    # aggregate raw results into a table
    raw_table = []
    for result in pickle_files:

        # load logger
        log = pd.read_pickle(result)

        # assign experiment name
        log['Data'] = result.split('generative_')[-1].split('_metrics')[0]

        # append experiment to results table
        raw_table.append(log)

    # concatenate and clean up table
    raw_table = pd.concat(raw_table)
    raw_table = raw_table.drop(['Entropy'], axis=1)
    raw_table = raw_table[raw_table.Method != 'EB-MAP-VAE']
    raw_table = raw_table[raw_table.Method != 'EB-V3AE-Gamma']
    if main_body:
        raw_table = raw_table[raw_table.BatchNorm == False]
        raw_table = raw_table.drop(['BatchNorm'], axis=1)

    return raw_table


def string_table(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: '{:.2f}'.format(x) if abs(x) > 0.1 else '{:.2e}'.format(x))
    return df


def generate_tables(results, bold_statistical_ties):

    # drop non-reported columns
    results = results.drop(['Best Epoch'], axis=1)

    # get number of trials
    n_trials = max(results.index) + 1

    # get means and standard deviations
    mean = pd.DataFrame(results.groupby(['Dataset', 'Method'], sort=False).mean())
    std = pd.DataFrame(results.groupby(['Dataset', 'Method'], sort=False).std(ddof=1))

    # build string table
    df = string_table(mean.copy(deep=True))
    if n_trials >= 2:
        df += '$\\pm$' + string_table(std.copy(deep=True))

    # loop over the metrics
    for metric in mean.columns:

        # get top performer
        i_best = np.argmax(mean[metric]) if metric == 'LL' else np.argmin(mean[metric])

        # bold winner
        df.loc[mean[metric].index[i_best], metric] = '\\textbf{' + df.loc[mean[metric].index[i_best], metric] + '}'

        # bold statistical ties if sufficient trials and requested
        if n_trials >= 2 and bold_statistical_ties:

            # get null hypothesis
            null_mean = mean[metric].to_numpy()[i_best]
            null_std = std[metric].to_numpy()[i_best]

            # compute p-values
            ms = zip([m for m in mean[metric].to_numpy().tolist()], [s for s in std[metric].to_numpy().tolist()])
            p = [ttest_ind_from_stats(null_mean, null_std, n_trials, m, s, n_trials, False)[-1] for (m, s) in ms]

            # bold statistical ties for best
            for i in range(df.shape[0]):
                if p[i] >= 0.05:
                    df.loc[mean[metric].index[i], metric] = '\\textbf{' + df.loc[mean[metric].index[i], metric] + '}'

    # # concatenate experiment to results table
    # if main_body:
    #     table = pd.concat([table, df.unstack(level=0).T.swaplevel(0, 1)])
    # else:
    #     table = pd.concat([table, df])

    return df.to_latex(escape=False)


def image_reshape(x):
    return np.reshape(tf.transpose(x, [1, 0, 2, 3]), [x.shape[1], -1, x.shape[-1]])


def generate_plots(pickle_files):

    # get raw results
    raw_table = raw_result_table(pickle_files, main_body=True)

    # loop over the experiments and methods
    for data in raw_table['Data'].unique():

        # load plot data
        with open(os.path.join('results', 'generative_' + data + '_plots.pkl'), 'rb') as f:
            plots = pickle.load(f)

        # get table and methods for this data set
        t_data = raw_table[raw_table.Data == data]
        methods = [m for m in t_data['Method'].unique() if m not in {'EB-MAP-VAE', 'EB-V3AE-Gamma'}]

        # grab original data
        x = np.squeeze(image_reshape(plots['x'][0::2]))

        # initialize figure
        fig, ax = plt.subplots(len(methods), 1, figsize=(16, 1.3 * len(methods)))
        plt.subplots_adjust(left=0.03, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

        # loop over the methods for this data set
        for i, method in enumerate(methods):

            # select between the better of batch normalization being on or off
            t_method = t_data[t_data.Method == method]
            best_row = np.argmin(t_method['RMSE'])
            batch_norm = False  # t_method.iloc[best_row]['BatchNorm']

            # grab the mean, std, and samples
            best_trial = t_method['RMSE'].idxmin()
            mean = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['mean'][0::2]))
            std = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['std'][0::2]))
            if len(std.shape) == 3:
                std = 1 - std
            sample = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['sample'][0::2]))
            ax[i].imshow(np.concatenate((x, mean, std, sample), axis=0), vmin=0, vmax=1, cmap='Greys')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_ylabel(method.replace('Uniform', 'MLE'), fontsize=13)

        # save figure
        fig.savefig(os.path.join('assets', 'fig_vae_samples_' + data + '.pdf'))


def generative_analysis():

    # experiment directory
    experiment_dir = os.path.join('results', 'vae')

    # load results for reach data set
    results = pd.DataFrame()
    for dataset in os.listdir(experiment_dir):
        logger = pd.DataFrame()
        for p in glob.glob(os.path.join(experiment_dir, dataset, '*_metrics.pkl')):
            logger = logger.append(pd.read_pickle(p))
        logger['Dataset'] = dataset.replace('_', ' ')
        results = results.append(logger)

    # build tables
    with open(os.path.join('assets', 'generative_table.tex'), 'w') as f:
        print(generate_tables(results, bold_statistical_ties=False), file=f)

    # generate plots
    generate_plots(results)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run analysis accordingly
    generative_analysis()

    # hold the plots
    plt.show()
