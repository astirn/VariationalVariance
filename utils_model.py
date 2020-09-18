import numpy as np
import tensorflow as tf


def softplus_inverse(x):
    """
    Inverse of the softplus function
    :param x: must be > 0
    :return: inverse-softplus(x)
    """
    return tf.math.log(tf.exp(x) - 1)


def expected_log_normal(x, mu, alpha, beta):
    """
    :param x: observations with leading batch dimension and where remaining dimensions constitute event shape
    :param mu: mean parameter with leading MC sample dimension followed by batch dimension
    :param alpha: precision shape parameter with leading MC sample dimension followed by batch dimension
    :param beta: precision scale parameter with leading MC sample dimension followed by batch dimension
    :return: E_{q(lambda | alpha, beta} [log Normal(x | mu, lambda)]
    """
    expected_lambda = alpha / beta
    expected_log_lambda = tf.math.digamma(alpha) - tf.math.log(beta)
    ll = 0.5 * (expected_log_lambda - tf.math.log(2 * np.pi) - (x - mu) ** 2 * expected_lambda)
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
