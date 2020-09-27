import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def softplus_inverse(x):
    """
    Inverse of the softplus function
    :param x: must be > 0
    :return: inverse-softplus(x)
    """
    return tf.math.log(tf.exp(x) - 1)


def expected_log_normal(x, mu, precision, log_precision):
    """
    :param x: observations where trailing dimension constitutes an event with diagonal covariance
    :param mu: mean parameter
    :param precision: precision parameter
    :param log_precision: log precision parameter
    :return: E_{q(lambda | alpha, beta} [log Normal(x | mu, lambda)]
    """
    ll = 0.5 * (log_precision - tf.math.log(2 * np.pi) - (x - mu) ** 2 * precision)
    return tf.reduce_sum(ll, axis=-1)


def student_log_prob(x, mu, alpha, beta):
    """
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#In_terms_of_inverse_scaling_parameter_%CE%BB
    :param x: observations with leading batch dimension and where remaining dimensions constitute event shape
    :param mu: mean parameter with leading MC sample dimension followed by batch dimension
    :param alpha: precision shape parameter with leading MC sample dimension followed by batch dimension
    :param beta: precision scale parameter with leading MC sample dimension followed by batch dimension
    :return: log Student(x | mu, alpha, beta)
    """
    nu = 2 * alpha
    lam = alpha / beta
    log_p = tf.math.lgamma(nu / 2 + 0.5) - tf.math.lgamma(nu / 2) + \
            0.5 * (tf.math.log(lam) - tf.math.log(np.pi) - tf.math.log(nu)) - \
            (nu / 2 + 0.5) * tf.math.log(1 + lam * (x - mu) ** 2 / nu)
    return tf.reduce_sum(log_p, axis=-1)


def mixture_proportions(archetype):
    """
    :param archetype: something of shape [num MC samples, batch size, event size]
    :return: a uniform categorical distribution over MC samples
    """
    return tfp.distributions.Categorical(logits=tf.transpose(tf.ones(tf.shape(archetype)[:2])))


def monte_carlo_student_t(mean, precision_samples):
    """
    :param mean: student's T location parameter
    :param precision_samples: precision samples
    :return: a uniform mixture of Normal distributions with common location but scales defined from precision samples
    """
    components = []
    for p in tf.unstack(precision_samples):
        normal = tfp.distributions.Normal(loc=mean, scale=p ** -0.5)
        components.append(tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1))
    return tfp.distributions.Mixture(cat=mixture_proportions(precision_samples), components=components)


class VariationalVariance(object):

    def __init__(self, dim_precision, prior_type, prior_fam, **kwargs):
        assert isinstance(dim_precision, int) and dim_precision > 0
        assert prior_type in {'MLE', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}
        assert prior_fam in {'Gamma', 'LogNormal'}

        # save configuration
        self.prior_type = prior_type
        self.prior_fam = prior_fam

        # configure prior
        if self.prior_type == 'Standard':
            a = tf.constant([kwargs.get('a')] * dim_precision, dtype=tf.float32)
            b = tf.constant([kwargs.get('b')] * dim_precision, dtype=tf.float32)
            self.pp = self.precision_prior(a, b)
        elif 'VAMP' in self.prior_type:
            # pseudo-inputs
            trainable = '*' in self.prior_type
            self.u = tf.Variable(initial_value=kwargs.get('u'), dtype=tf.float32, trainable=trainable, name='u')
        elif self.prior_type == 'VBEM':
            # fixed prior parameters for precision
            params = [0.05, 0.1, 0.25, 0.5, 0.75, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            if self.prior_fam == 'Gamma':
                uv = softplus_inverse(np.array(tuple(itertools.product(params, params)), dtype=np.float32).T)
                u = tf.expand_dims(uv[0], axis=-1)
                v = tf.expand_dims(uv[1], axis=-1)
            else:
                uv = np.array(tuple(itertools.product(params, params)), dtype=np.float32).T
                mean = uv[0] / uv[1]
                var = uv[0] / uv[1] ** 2
                u = tf.expand_dims(tf.math.log(mean ** 2 / (mean ** 2 + var) ** 0.5), axis=-1)
                v = softplus_inverse(tf.expand_dims(tf.math.log(1 + var / (mean ** 2)), axis=-1))
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=False, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=False, name='v')
        elif self.prior_type == 'VBEM*':
            # trainable prior parameters for precision
            k = kwargs.get('k')
            u = tf.random.uniform(shape=(k, dim_precision), minval=-3, maxval=3, dtype=tf.float32)
            v = tf.random.uniform(shape=(k, dim_precision), minval=-3, maxval=3, dtype=tf.float32)
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

    def expected_precision(self, alpha, beta):
        """
        :param alpha: precision shape (Gamma distributed precision) or mean (LogNormal distributed precision) parameter
        :param beta: precision scale parameter with leading MC sample dimension followed by batch dimension
        :return: E_{q(lambda | alpha, beta} [lambda]
        """
        if self.prior_fam == 'Gamma':
            return alpha / beta
        else:  # self.prior_fam == 'LogNormal':
            return tf.exp(alpha + beta ** 2 / 2)

    def expected_log_precision(self, alpha, beta):
        """
        :param alpha: precision shape (Gamma distributed precision) or mean (LogNormal distributed precision) parameter
        :param beta: precision scale parameter with leading MC sample dimension followed by batch dimension
        :return: E_{q(lambda | alpha, beta} [log lambda]
        """
        if self.prior_fam == 'Gamma':
            return tf.math.digamma(alpha) - tf.math.log(beta)
        else:  # self.prior_fam == 'LogNormal':
            return alpha

    def precision_prior(self, alpha, beta):
        if self.prior_fam == 'Gamma':
            prior = tfp.distributions.Gamma(alpha, beta)
        else:  # self.prior_fam == 'LogNormal':
            prior = tfp.distributions.LogNormal(alpha, beta)
        return tfp.distributions.Independent(prior, reinterpreted_batch_ndims=1)

    def variational_precision(self, alpha, beta, leading_mc_dimension):
        if self.prior_fam == 'Gamma':
            qp = tfp.distributions.Gamma(alpha, beta)
        else:  # self.prior_fam == 'LogNormal':
            qp = tfp.distributions.LogNormal(alpha, beta)
        qp = tfp.distributions.Independent(qp, reinterpreted_batch_ndims=1)
        p_samples = qp.sample(sample_shape=() if leading_mc_dimension else self.num_mc_samples)
        return qp, p_samples

    def dkl_precision(self, qp, p_samples, pi_parent_samples, vamp_samples=None):
        assert not ('VAMP' in self.prior_type and vamp_samples is None)

        # compute kl-divergence depending on prior type
        if self.prior_type == 'Standard':
            dkl = qp.kl_divergence(self.pp)
        elif 'VAMP' in self.prior_type or 'VBEM' in self.prior_type:

            # prior's mixture proportions--shape will be [# components, batch size (or 1), # MC samples (or 1)]
            if self.prior_type in {'VAMP', 'VAMP*'}:
                pi = tf.ones(self.u.shape[0]) / self.u.shape[0]
                pi = tf.reshape(pi, [-1, 1, 1])
            else:
                pi = tf.vectorized_map(lambda s: self.pi(s), pi_parent_samples)
                pi = tf.transpose(pi, [2, 1, 0])

            # prior's mixture components--shape will be [# components, # MC samples, event shape]
            if 'VAMP' in self.prior_type:
                alpha = tf.transpose(tf.vectorized_map(lambda u: self.alpha(u), vamp_samples), [1, 0, 2])
                beta = tf.transpose(tf.vectorized_map(lambda u: self.beta(u), vamp_samples), [1, 0, 2])
            else:
                alpha = tf.expand_dims(tf.nn.softplus(self.u), axis=1)
                beta = tf.expand_dims(tf.nn.softplus(self.v), axis=1)

            # clip precision samples to avoid 0 which has infinite density for some parameter values
            p_samples = tf.clip_by_value(p_samples, clip_value_min=1e-30, clip_value_max=tf.float32.max)

            # reshape precision samples to [batch shape, num MC samples, event shape]
            p_samples = tf.transpose(p_samples, [1, 0, 2])

            # compute prior log probabilities for each component--shape is [# components, batch size, # MC samples]
            log_pp_c = tf.vectorized_map(lambda ab: self.precision_prior(ab[0], ab[1]).log_prob(p_samples),
                                         elems=(alpha, beta))

            # take the expectation w.r.t. to mixture proportions--shape will be [# MC samples, batch size]
            epsilon = 1e-30
            log_pp = tf.reduce_logsumexp(tf.math.log(pi + epsilon) + log_pp_c, axis=0)  # add offset to avoid log(0)
            log_pp -= tf.math.log(tf.reduce_sum(epsilon + pi, axis=0))  # correct for the offset
            log_pp = tf.transpose(log_pp)

            # average over MC samples--shape will be [batch size]
            dkl = tf.reduce_mean(-qp.entropy() - log_pp, axis=0)

        else:
            dkl = tf.constant(0.0, dtype=tf.float32)

        return dkl

