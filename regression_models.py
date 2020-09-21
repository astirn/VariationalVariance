import os
import argparse
import itertools
import numpy as np
import tensorflow as tf
import scipy.stats as sps
import tensorflow_probability as tfp

import seaborn as sns
from matplotlib import pyplot as plt

from utils_model import softplus_inverse, mixture_proportions, VariationalVariance
from callbacks import RegressionCallback
from regression_data import generate_toy_data

# workaround: https://github.com/tensorflow/tensorflow/issues/34888
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name=None):
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    nn.add(tf.keras.layers.Dense(d_hidden, f_hidden))
    nn.add(tf.keras.layers.Dense(d_out, f_out))
    return nn


class LocationScaleRegression(tf.keras.Model):

    def __init__(self, y_mean, y_var):
        super(LocationScaleRegression, self).__init__()

        # save configuration
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)

    def whiten_targets(self, y):
        return (y - self.y_mean) / self.y_std

    def de_whiten_mean(self, mu):
        return mu * self.y_std + self.y_mean

    def de_whiten_stddev(self, sigma):
        return sigma * self.y_std

    def de_whiten_precision(self, precision):
        return precision / self.y_var

    def de_whiten_log_precision(self, log_precision):
        return log_precision - tf.math.log(self.y_var)

    def call(self, inputs, **kwargs):
        self.objective(x=inputs['x'], y=inputs['y'])
        return tf.constant(0.0, dtype=tf.float32)


class NormalRegression(LocationScaleRegression):

    def __init__(self, d_in, d_hidden, f_hidden, d_out, y_mean, y_var, **kwargs):
        super(NormalRegression, self).__init__(y_mean, y_var)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0

        # build parameter networks
        self.mean = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        self.precision = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softplus', name='lambda')

    def ll(self, y, mean, precision, whiten_targets):
        if whiten_targets:
            y = self.whiten_targets(y)
        else:
            mean = self.de_whiten_mean(mean)
            precision = self.de_whiten_precision(precision)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5).log_prob(y)

    def objective(self, x, y):

        # run parameter networks
        mean = self.mean(x)
        precision = self.precision(x)

        # compute log likelihood on whitened targets
        ll_whitened = self.ll(y, mean, precision, whiten_targets=True)

        # use negative log likelihood on whitened targets as minimization objective
        self.add_loss(-tf.reduce_mean(ll_whitened))

        # compute de-whitened performance
        ll = self.ll(y, mean, precision, whiten_targets=False)
        rmse = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y, self.de_whiten_mean(mean)), axis=-1))

        # assign model's log likelihood (Bayesian methods will use log posterior predictive likelihood)
        ll_model = ll

        # additional observation metrics
        self.add_metric(ll_whitened, name='LL (whitened)', aggregation='mean')
        self.add_metric(ll, name='LL', aggregation='mean')
        self.add_metric(ll_model, name='Model LL', aggregation='mean')
        self.add_metric(rmse, name='RMSE', aggregation='mean')

    def model_mean(self, x):
        """Model mean is simply the mean network's output trained using maximum likelihood"""
        return self.de_whiten_mean(self.mean(x))

    def model_stddev(self, x):
        """Model standard dev. is simply the transformed precision network's output trained using maximum likelihood"""
        return self.de_whiten_stddev(self.precision(x) ** -0.5)


class VariationalPrecisionNormalRegression(tf.keras.Model, VariationalVariance):

    def __init__(self, d_in, d_hidden, f_hidden, d_out, prior_type, prior_fam, y_mean, y_var, n_mc=1, **kwargs):
        tf.keras.Model.__init__(self)
        VariationalVariance.__init__(self, d_out, prior_type, prior_fam, **kwargs)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0
        # assert prior_type in {'MLE', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}
        # assert prior_fam in {'Gamma', 'LogNormal'}
        assert isinstance(n_mc, int) and n_mc > 0

        # save configuration
        # self.prior_type = prior_type
        # self.prior_fam = prior_fam
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)
        self.num_mc_samples = n_mc

        # # configure prior
        # if self.prior_type == 'Standard':
        #     a = tf.constant([kwargs.get('a')] * d_out, dtype=tf.float32)
        #     b = tf.constant([kwargs.get('b')] * d_out, dtype=tf.float32)
        #     self.pp = self.precision_prior(a, b)
        # elif 'VAMP' in self.prior_type:
        #     # pseudo-inputs
        #     trainable = '*' in self.prior_type
        #     self.u = tf.Variable(initial_value=kwargs.get('u'), dtype=tf.float32, trainable=trainable, name='u')
        # elif self.prior_type == 'VBEM':
        #     # fixed prior parameters for precision
        #     params = [0.05, 0.1, 0.25, 0.5, 0.75, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        #     if self.prior_fam == 'Gamma':
        #         uv = softplus_inverse(np.array(tuple(itertools.product(params, params)), dtype=np.float32).T)
        #         u = tf.expand_dims(uv[0], axis=-1)
        #         v = tf.expand_dims(uv[1], axis=-1)
        #     else:
        #         uv = np.array(tuple(itertools.product(params, params)), dtype=np.float32).T
        #         mean = uv[0] / uv[1]
        #         var = uv[0] / uv[1] ** 2
        #         u = tf.expand_dims(tf.math.log(mean ** 2 / (mean ** 2 + var) ** 0.5), axis=-1)
        #         v = softplus_inverse(tf.expand_dims(tf.math.log(1 + var / (mean ** 2)), axis=-1))
        #     self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=False, name='u')
        #     self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=False, name='v')
        # elif self.prior_type == 'VBEM*':
        #     # trainable prior parameters for precision
        #     k = kwargs.get('k')
        #     u = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
        #     v = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
        #     self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
        #     self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

        # build parameter networks
        self.mu = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        alpha_f_out = 'softplus' if self.prior_fam == 'Gamma' else None
        self.alpha = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=alpha_f_out, name='alpha')
        self.beta = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softplus', name='beta')
        if self.prior_type in {'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}:
            self.pi = neural_network(d_in, d_hidden, f_hidden, self.u.shape[0], f_out='softmax', name='pi')

    # def precision_prior(self, alpha, beta):
    #     if self.prior_fam == 'Gamma':
    #         prior = tfp.distributions.Gamma(alpha, beta)
    #     elif self.prior_fam == 'LogNormal':
    #         prior = tfp.distributions.LogNormal(alpha, beta)
    #     return tfp.distributions.Independent(prior, reinterpreted_batch_ndims=1)
    #
    # def qp(self, x):
    #     if self.prior_fam == 'Gamma':
    #         qp = tfp.distributions.Gamma(self.alpha(x), self.beta(x))
    #     elif self.prior_fam == 'LogNormal':
    #         qp = tfp.distributions.LogNormal(self.alpha(x), self.beta(x))
    #     return tfp.distributions.Independent(qp)
    #
    # def variational_family(self, x):
    #     # variational family q(precision|x)
    #     qp = self.qp(x)
    #
    #     # compute kl-divergence depending on prior type
    #     if self.prior_type == 'Standard':
    #         dkl = qp.kl_divergence(self.pp)
    #     elif 'VAMP' in self.prior_type or 'VBEM' in self.prior_type:
    #
    #         # compute prior's mixture proportions
    #         pi = tf.ones(self.u.shape[0]) / self.u.shape[0] if self.prior_type in {'VAMP', 'VAMP*'} else self.pi(x)
    #
    #         # compute prior's mixture components
    #         if 'VAMP' in self.prior_type:
    #             alpha = self.alpha(self.u)
    #             beta = self.beta(self.u)
    #         else:
    #             alpha = tf.nn.softplus(self.u) if self.prior_fam == 'Gamma' else self.u
    #             beta = tf.nn.softplus(self.v)
    #         pp_c = self.precision_prior(alpha, beta)
    #
    #         # MC estimate kl-divergence due to pesky log-sum
    #         p_samples = qp.sample(self.num_mc_samples)
    #         p_samples = tf.tile(tf.expand_dims(p_samples, axis=-2), [1, 1] + pp_c.batch_shape.as_list() + [1])
    #         log_pi = tf.math.log(tf.expand_dims(pi, axis=0))
    #         log_pp_c = tf.clip_by_value(pp_c.log_prob(p_samples), clip_value_min=tf.float32.min, clip_value_max=100)
    #         log_pp = tf.reduce_logsumexp(log_pi + log_pp_c, axis=-1)
    #         dkl = -qp.entropy() - tf.reduce_mean(log_pp, axis=0)
    #
    #     else:
    #         dkl = tf.constant(0.0)
    #
    #     return qp, dkl

    def expected_log_lambda(self, alpha, beta):
        if self.prior_fam == 'Gamma':
            return tf.math.digamma(alpha) - tf.math.log(beta)
        elif self.prior_fam == 'LogNormal':
            return alpha

    @ staticmethod
    def ll(y, mu, expected_lambda, expected_log_lambda):
        ll = 0.5 * (expected_log_lambda - tf.math.log(2 * np.pi) - (y - mu) ** 2 * expected_lambda)
        return tf.reduce_sum(ll, axis=-1)

    def whiten(self, y, mu, expected_lambda, expected_log_lambda):
        y = (y - self.y_mean) / self.y_std
        return y, mu, expected_lambda, expected_log_lambda

    def de_whiten(self, y, mu, expected_lambda, expected_log_lambda):
        mu = mu * self.y_std + self.y_mean
        expected_lambda = expected_lambda / self.y_var
        expected_log_lambda = expected_log_lambda - tf.math.log(self.y_var)
        return y, mu, expected_lambda, expected_log_lambda

    def variational_objective(self, x, y):

        # run parameter networks
        mu = self.mu(x)
        alpha = self.alpha(x)
        beta = self.beta(x)

        # variational family
        qp, p_samples = self.variational_precision(alpha, beta, leading_mc_dimension=False)

        # variational variance log likelihood E_{q(lambda|alpha(x), beta(x))}[log p(y|mu(x), lambda)]
        expected_log_lambda = self.expected_log_lambda(alpha, beta)
        ell = self.ll(*self.whiten(y, mu, qp.mean(), expected_log_lambda))

        # compute KL divergence w.r.t. p(lambda)
        vamp_samples = tf.expand_dims(self.u, axis=0) if 'VAMP' in self.prior_type else None
        dkl = self.dkl_precision(qp, p_samples, pi_parent_samples=tf.expand_dims(x, axis=0), vamp_samples=vamp_samples)

        # evidence lower bound
        elbo = ell - dkl

        # compute adjusted log likelihood of non-scaled y using de-whitened model parameter
        ll_adjusted = self.ll(*self.de_whiten(y, mu, qp.mean(), expected_log_lambda))

        # compute squared error for reporting purposes
        error_dist = tf.norm(y - (mu * self.y_std + self.y_mean), axis=-1)
        squared_error = error_dist ** 2

        # add metrics for call backs
        self.add_metric(elbo, name='ELBO', aggregation='mean')
        self.add_metric(ell, name='ELL', aggregation='mean')
        self.add_metric(dkl, name='KL', aggregation='mean')
        self.add_metric(ll_adjusted, name='ELL (adjusted)', aggregation='mean')
        self.add_metric(error_dist, name='MAE', aggregation='mean')
        self.add_metric(squared_error, name='MSE', aggregation='mean')

        # add minimization objective
        self.add_loss(-tf.reduce_mean(elbo))

        # add log posterior predictive likelihood
        py_x = self.posterior_predictive(mu, alpha, beta, p_samples, de_whiten=False)
        self.add_metric(py_x.log_prob(y), name='LPPL', aggregation='mean')
        py_x = self.posterior_predictive(mu, alpha, beta, p_samples, de_whiten=True)
        self.add_metric(py_x.log_prob(y), name='LPPL (adjusted)', aggregation='mean')

    def posterior_predictive(self, mu, alpha, beta, p_samples, de_whiten=False):
        shift = self.y_mean if de_whiten else 0.0
        scale = self.y_std if de_whiten else 1.0
        if self.prior_fam == 'Gamma':
            py_x = tfp.distributions.StudentT(df=2 * alpha, loc=mu * scale + shift, scale=tf.sqrt(beta / alpha) * scale)
            return tfp.distributions.Independent(py_x, reinterpreted_batch_ndims=1)
        elif self.prior_fam == 'LogNormal':
            components = []
            for p in tf.unstack(p_samples):
                p = tfp.distributions.Normal(loc=mu * scale + shift, scale=p ** -0.5 * scale)
                components.append(tfp.distributions.Independent(p, reinterpreted_batch_ndims=1))
            py_x = tfp.distributions.Mixture(cat=mixture_proportions(p_samples), components=components)
            return py_x

    def posterior_mean(self, x):
        return self.mu(x) * self.y_std + self.y_mean

    def posterior_stddev(self, x):
        if self.prior_fam == 'Gamma':
            alpha = self.alpha(x)
            beta = self.beta(x)
            return tf.exp(tf.math.lgamma(alpha - 0.5)) / tf.exp(tf.math.lgamma(alpha)) * tf.sqrt(beta) * self.y_std
        elif self.prior_fam == 'LogNormal':
            return tf.exp(self.beta(x) ** 2 / 8 - self.alpha(x) / 2)

    def call(self, inputs, **kwargs):
        self.variational_objective(x=inputs['x'], y=inputs['y'])
        return tf.constant(0.0, dtype=tf.float32)


def prior_params(precisions, prior_fam):
    if prior_fam == 'Gamma':
        a, _, b_inv = sps.gamma.fit(precisions, floc=0)
        b = 1 / b_inv
    else:
        a, b = np.mean(np.log(precisions)), np.std(np.log(precisions))
    print(prior_fam, 'Prior:', a, b)
    return a, b


def fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_eval = np.squeeze(x_eval)
    true_mean = np.squeeze(true_mean)
    true_std = np.squeeze(true_std)
    mdl_mean = np.squeeze(mdl_mean)
    mdl_std = np.squeeze(mdl_std)

    # get a new figure
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)

    # plot the data
    sns.scatterplot(x_train, y_train, ax=ax[0])

    # plot the true mean and standard deviation
    ax[0].plot(x_eval, true_mean, '--k')
    ax[0].plot(x_eval, true_mean + true_std, ':k')
    ax[0].plot(x_eval, true_mean - true_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_eval, mdl_mean)[0]
    ax[0].fill_between(x_eval[:, ], mdl_mean - mdl_std, mdl_mean + mdl_std, color=l.get_color(), alpha=0.5)
    ax[0].plot(x_eval, true_mean, '--k')

    # clean it up
    ax[0].set_ylim([-20, 20])
    ax[0].set_ylabel('y')

    # plot the std
    ax[1].plot(x_eval, mdl_std, label='predicted')
    ax[1].plot(x_eval, true_std, '--k', label='truth')
    ax[1].set_ylim([0, 5])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('std(y|x)')
    plt.legend()

    return fig


if __name__ == '__main__':

    # enable background tiles on plots
    sns.set(color_codes=True)

    # unit test
    test = np.random.uniform(-10, 10, 100)
    assert (np.abs(softplus_inverse(tf.nn.softplus(test)) - test) < 1e-6).all()
    test = np.random.uniform(0, 10, 100)
    assert (np.abs(tf.nn.softplus(softplus_inverse(test)) - test) < 1e-6).all()

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='Normal', help='algorithm')
    parser.add_argument('--prior_type', default='xVAMP*', type=str, help='prior type')
    parser.add_argument('--seed', default=1234, type=int, help='prior type')
    args = parser.parse_args()

    # random number seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # set configuration
    D_HIDDEN = 50
    PRIOR_TYPE = args.prior_type
    PRIOR_FAM = 'Gamma' if 'Gamma' in args.algorithm else 'LogNormal'
    N_MC_SAMPLES = 50
    LEARNING_RATE = 1e-2
    CLIP_VALUE = None if args.algorithm == 'Normal' else 0.5
    EPOCHS = int(6e3)

    # load data
    x_train, y_train, x_eval, true_mean, true_std = generate_toy_data()
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])

    # compute standard prior according to prior family
    A, B = prior_params(1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2, PRIOR_FAM)

    # VAMP prior pseudo-input initializers
    U = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), 20), axis=-1)

    # pick the appropriate model
    MODEL = NormalRegression if args.algorithm == 'Normal' else VariationalPrecisionNormalRegression

    # initialize model instance
    mdl = MODEL(d_in=x_train.shape[1],
                d_hidden=D_HIDDEN,
                f_hidden='sigmoid',
                d_out=y_train.shape[1],
                y_mean=0.0,
                y_var=1.0,
                a=A,
                b=B,
                k=20,
                u=U,
                n_mc=N_MC_SAMPLES)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=CLIP_VALUE)
    mdl.compile(optimizer=optimizer, loss=[None], run_eagerly=False)

    # train model
    hist = mdl.fit(ds_train, epochs=EPOCHS, verbose=0, callbacks=[RegressionCallback(EPOCHS)])

    # plot results for toy data
    mdl_mean, mdl_std = mdl.model_mean(x_eval), mdl.model_stddev(x_eval)
    fig = plt.figure()
    fig.suptitle(args.algorithm)
    plt.plot(hist.history['Model LL'])
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.algorithm)
    plt.show()
