import os
import glob
import argparse
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from regression_data import generate_toy_data, REGRESSION_DATA
from regression_experiments_v2 import RESULTS_DIR
from utils_analysis import make_clean_method_names, build_table, champions_club_table

# enable background tiles on plots
sns.set(color_codes=True)


def query_string(alg, prior):
    return "Algorithm == '" + alg + "' and Prior == '" + prior + "'"


def regression_subplot(alg, prior, ll_logger, data_logger, mv_logger, ax, color):
    # get best performance for this algorithm/prior combination
    i_best = ll_logger.query(query_string(alg, prior))['LL'].idxmax()

    # plot the training data
    data = data_logger.query(query_string(alg, prior)).loc[i_best]
    sns.scatterplot(data['x'], data['y'], ax=ax, color=color)

    # plot the model's mean and standard deviation
    model = mv_logger.query(query_string(alg, prior)).loc[i_best]
    ax.plot(model['x'], model['mean(y|x)'], color=color)
    ax.fill_between(model['x'],
                    model['mean(y|x)'] - 2 * model['std(y|x)'],
                    model['mean(y|x)'] + 2 * model['std(y|x)'],
                    color=color, alpha=0.5)

    # plot the true mean and standard deviation
    _, _, x_eval, true_mean, true_std = generate_toy_data()
    ax.plot(x_eval, true_mean, '--k')
    ax.plot(x_eval, true_mean + 2 * true_std, ':k')
    ax.plot(x_eval, true_mean - 2 * true_std, ':k')

    # make it pretty
    ax.set_title(model['Method'].unique()[0])


def toy_regression_plot(ll_logger, data_logger, mv_logger):
    # make clean method names for report
    ll_logger = make_clean_method_names(ll_logger)
    data_logger = make_clean_method_names(data_logger)
    mv_logger = make_clean_method_names(mv_logger)

    # get priors
    priors = ll_logger['Prior'].unique()

    # size toy data figure
    n_rows, n_cols = 3, len(priors)
    fig = plt.figure(figsize=(2.9 * n_cols, 2.9 * n_rows), constrained_layout=False)
    gs = fig.add_gridspec(n_rows, n_cols)
    for i, prior in enumerate(priors):
        fig.add_subplot(gs[0, i])
        fig.add_subplot(gs[1, i])
        fig.add_subplot(gs[2, i])

    # make it tight
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.95, wspace=0.15, hspace=0.15)

    # get color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # plot toy regression subplots
    i = -1
    for prior in ['N/A', 'MLE', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*']:
        if prior not in priors:
            continue
        i += 1

        # first row subplots
        ax = fig.axes[n_rows * i]
        alg1 = 'Detlefsen' if prior == 'N/A' else 'Gamma-Normal'
        regression_subplot(alg1, prior, ll_logger, data_logger, mv_logger, ax, colors[0])
        ax.set_xlim([-5, 15])
        ax.set_ylim([-25, 25])
        ax.set_xlabel('')
        ax.set_xticklabels([])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # second row subplots
        ax = fig.axes[n_rows * i + 1]
        alg2 = 'Detlefsen (fixed)' if prior == 'N/A' else 'LogNormal-Normal'
        regression_subplot(alg2, prior, ll_logger, data_logger, mv_logger, ax, colors[1])
        ax.set_xlim([-5, 15])
        ax.set_ylim([-25, 25])
        ax.set_xlabel('')
        ax.set_xticklabels([])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # third row subplots
        ax = fig.axes[n_rows * i + 2]
        _, _, x_eval, _, true_std = generate_toy_data()
        ax.plot(x_eval, true_std, 'k', label='truth')
        data = mv_logger.query(query_string(alg1, prior))
        data = data.append(mv_logger.query(query_string(alg2, prior)))
        sns.lineplot(x='x', y='std(y|x)', hue='Method', ci='sd', data=data, ax=ax)
        ax.legend().remove()
        ax.set_xlim([-5, 15])
        ax.set_ylim([0, 6])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    return fig


def toy_regression_analysis():
    # get all the pickle files
    data_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', 'toy', '*_data.pkl')))
    mv_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', 'toy', '*_mv.pkl')))
    prior_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', 'toy', '*_prior.pkl')))
    ll_pickles = set(glob.glob(os.path.join(RESULTS_DIR, '*', 'toy', '*.pkl'))) - \
                 data_pickles.union(mv_pickles, prior_pickles)

    # aggregate results into single data frame
    ll_logger = pd.DataFrame()
    for p in ll_pickles:
        ll_logger = ll_logger.append(pd.read_pickle(p))
    data_logger = pd.DataFrame()
    for p in data_pickles:
        data_logger = data_logger.append(pd.read_pickle(p))
    mv_logger = pd.DataFrame()
    for p in mv_pickles:
        mv_logger = mv_logger.append(pd.read_pickle(p))

    # generate plot
    fig = toy_regression_plot(ll_logger, data_logger, mv_logger)
    fig.savefig(os.path.join('assets', 'fig_toy.png'))
    fig.savefig(os.path.join('assets', 'fig_toy.pdf'))


def exclude_log_normal(df, **kwargs):
    return df[df.Algorithm != 'LogNormal-Normal']


def uci_regression_analysis():

        # experiment directory
        experiment_dir = os.path.join(RESULTS_DIR, 'regression_uci_{:d}'.format(iterations))

    # load results for each data set
    results = dict()
    for dataset in REGRESSION_DATA.keys():
        result_dir = os.path.join(experiment_dir, dataset)
        if os.path.exists(result_dir):
            logger = pd.DataFrame()
            for p in glob.glob(os.path.join(result_dir, '*.pkl')):
                if '_prior' in p:
                    continue
                logger = logger.append(pd.read_pickle(p))
            results.update({dataset: logger})

    # make latex tables
    max_cols = 5
    with open(os.path.join('assets', 'regression_uci_ll.tex'), 'w') as f:
        table, ll_cc = build_table(results, 'LL', 'max', max_cols, bold_statistical_ties=False)
        print(table, file=f)
    with open(os.path.join('assets', 'regression_uci_rmse.tex'), 'w') as f:
        table, rmse_cc = build_table(results, 'Mean RMSE', 'min', max_cols, bold_statistical_ties=False)
        print(table, file=f)

    # print champions club
    with open(os.path.join('assets', 'regression_uci_champions_club.tex'), 'w') as f:
        print(champions_club_table([ll_cc, rmse_cc]), file=f)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='uci', help='experiment in {toy, uci}')
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run experiments accordingly
    if args.experiment == 'toy':
        toy_regression_analysis()
    else:
        uci_regression_analysis()

    # hold the plots
    plt.show()
