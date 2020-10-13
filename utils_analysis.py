import os
import pickle
import pandas as pd
from scipy.stats import ks_2samp


def make_clean_method_names(df):
    """
    Cleans prior names and adds a Method column from which plot labels can be created
    :param df: a pandas data frame containing experimental results
    :return: a pandas data frame containing the same results but with cleaner prior names and new methods column
    """
    # make clean method names for report
    df['Method'] = df['Algorithm'] + df['Prior'].apply(lambda s: '' if s == 'N/A' else ' (' + s + ')')
    return df


def string_table(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: '{:.2f}'.format(x) if 0.01 < abs(x) < 100 else '{:.1e}'.format(x))
    return df


def organize_regression_table(table):
    """
    :param table: data frame to organize
    :return: a data frame in (Algorithm, Prior) indices sorted according to desired manuscript ordering
    """
    table = table.reset_index()
    table.Algorithm = pd.Categorical(table.Algorithm, categories=['Detlefsen', 'Normal', 'Student', 'Gamma-Normal'])
    table.Prior = pd.Categorical(table.Prior, categories=['N/A', 'VAP', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'])
    table = table.sort_values(['Algorithm', 'Prior'])
    return table.set_index(keys=['Algorithm', 'Prior']).sort_index()


def build_table(results, metric, order, max_cols, bold_statistical_ties, process_fn=None, transpose=False):
    """
    :param results: dictionary of Panda data frames
    :param metric: name of desired metric (must be column in pandas data frame)
    :param order: how to order best results
    :param max_cols: max columns per row
    :param bold_statistical_ties: whether to bold statistical ties for first place
    :param process_fn: optional processing functions
    :param transpose: whether to transpose the table
    :return: 
    """
    if process_fn is None:
        process_fn = []
    assert order in {'max', 'min', None}

    # initialize champions club
    champions_club = None

    # aggregate results into table
    table = None
    test_table = None
    for exp, log in results.items():

        # load logger
        n_trials = max(log.index) + 1
        if n_trials < 2:
            continue

        # apply processing functions
        for fn in process_fn:
            log = fn(log, **{'mode': 'tex', 'metric': metric})

        # compute means and standard deviations over methods
        log = log.convert_dtypes()
        mean = pd.DataFrame(log.groupby(['Algorithm', 'Prior'], sort=False)[metric].mean())
        mean = mean.rename(columns={metric: exp}).sort_values(['Algorithm', 'Prior'])
        std = pd.DataFrame(log.groupby(['Algorithm', 'Prior'], sort=False)[metric].std(ddof=1))
        std = std.rename(columns={metric: exp}).sort_values(['Algorithm', 'Prior'])
        log = log.set_index(keys=['Algorithm', 'Prior']).sort_index()

        # initialize champions club if needed
        if champions_club is None:
            champions_club = mean.copy(deep=True)
            champions_club[metric + ' Hard Wins'] = 0
            champions_club[metric + ' Soft Wins'] = 0
            del champions_club[exp]

        # build table
        df = string_table(mean.copy(deep=True)) + '$\\pm$' + string_table(std.copy(deep=True))

        # get index of top performer, using numpy arg min/max is ok since only one dataset in mean table
        i_best = mean.idxmax(skipna=True) if order == 'max' else mean.abs().idxmin()

        # bold winner and update hard wins count
        df.loc[i_best] = '\\textbf{' + df.loc[i_best] + '}'
        champions_club.loc[i_best, metric + ' Hard Wins'] += 1

        # get null hypothesis
        null_samples = log.loc[i_best][metric].to_numpy()

        # compute p-values
        p = [ks_2samp(null_samples, log.loc[i][metric].to_numpy(), mode='exact')[-1] for i in log.index.unique()]

        # look for statistical ties
        if order is not None:
            for i, index in enumerate(log.index.unique()):
                if p[i] >= 0.05:
                    # update soft wins count
                    champions_club.loc[index, metric + ' Soft Wins'] += 1

                    # bold statistical ties if requested
                    if bold_statistical_ties:
                        df.loc[index] = '\\textbf{' + df.loc[index] + '}'

        # append experiment to results table
        table = df if table is None else table.join(df)

        # build test table for viewing with PyCharm SciView
        test_table = mean if test_table is None else test_table.join(mean)

    # make the table pretty
    table = organize_regression_table(table)
    if transpose:
        return table.T.to_latex(escape=False)

    # split tables into a maximum number of cols
    i = 0
    tables = []
    experiments = []
    while i < table.shape[1]:
        experiments.append(table.columns[i:i + max_cols])
        tables.append(table[experiments[-1]])
        i += max_cols
    tables = [t.to_latex(escape=False) for t in tables]

    # add experimental details
    for i in range(len(tables)):
        target = 'Algorithm & Prior'
        i_start = tables[i].find(target)
        i_stop = tables[i][i_start:].find('\\')
        assert len(tables[i][i_start + len(target):i_start + i_stop].split('&')) == len(experiments[i]) + 1
        details = ''
        for experiment in experiments[i]:
            experiment = experiment.split('_')[0]
            with open(os.path.join('data', experiment, experiment + '.pkl'), 'rb') as f:
                dd = pickle.load(f)
            details += '& ({:d}, {:d}, {:d})'.format(dd['data'].shape[0], dd['data'].shape[1], dd['target'].shape[1])
        tables[i] = tables[i][:i_start + len(target)] + details + tables[i][i_start + i_stop:]

    # merge the tables into a single table
    if len(tables) > 1:
        tables[0] = tables[0].split('\\bottomrule')[0]
    for i in range(1, len(tables)):
        tables[i] = '\\midrule' + tables[i].split('\\toprule')[-1]

    return ''.join(tables), champions_club


def champions_club_table(champion_clubs):
    """
    :param champion_clubs: a list of champion club data frames
    :return:
    """
    assert isinstance(champion_clubs, list)
    champions_club = pd.concat(champion_clubs, axis=1)
    champions_club_condensed = pd.DataFrame()
    columns = champions_club.columns
    for i in range(0, len(columns), 2):
        wins = [max(champions_club[columns[i]]), max(champions_club[columns[i + 1]])]
        champions_club[columns[i]] = champions_club[columns[i]].astype('str')
        champions_club[columns[i + 1]] = champions_club[columns[i + 1]].astype('str')
        for j in range(2):
            for index, row in champions_club.iterrows():
                if row[columns[i + j]] == str(wins[j]):
                    champions_club.loc[index, columns[i + j]] = '\\textbf{' + row[columns[i + j]] + '}'
                else:
                    champions_club.loc[index, columns[i + j]] = row[columns[i + j]]
        champions_club_condensed[columns[i].replace(' Hard Wins', '')] = \
            champions_club[columns[i]] + ' (' + champions_club[columns[i + 1]] + ')'

    return organize_regression_table(champions_club_condensed).to_latex(escape=False)
