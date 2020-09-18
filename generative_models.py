import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from scipy.stats import gamma

from generative_data import load_data_set
from utils_model import softplus_inverse, expected_log_normal
from callbacks import LearningCurveCallback, ReconstructionCallback, LatentVisualizationCallback2D

# workaround: https://github.com/tensorflow/tensorflow/issues/34888
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

EPSILON = 1e-6


def encoder_dense(dim_in, dim_out, batch_norm, name):
    enc = tf.keras.Sequential(name=name)
    enc.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    enc.add(tf.keras.layers.Flatten())
    enc.add(tf.keras.layers.Dense(units=512))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=256))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=128))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=dim_out))
    return enc


def decoder_dense(dim_in, dim_out, batch_norm, final_activation, name):
    dec = tf.keras.Sequential(name=name)
    dec.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    dec.add(tf.keras.layers.Flatten())
    dec.add(tf.keras.layers.Dense(units=128))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=256))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=512))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=dim_out, activation=final_activation))
    return dec


def encoder_convolution(dim_in, dim_out, _, name):
    return tf.keras.Sequential(name=name, layers=[
        tf.keras.Input(shape=dim_in, dtype=tf.float32),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.ELU(),
        tf.keras.layers.Dense(units=dim_out)])


def decoder_convolution(dim_in, dim_out, _, final_activation, name):
    return tf.keras.Sequential(name=name, layers=[
        tf.keras.Input(shape=dim_in, dtype=tf.float32),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.ELU(),
        tf.keras.layers.Dense(units=64 * dim_out[0] // 4 * dim_out[1] // 4),
        tf.keras.layers.ELU(),
        tf.keras.layers.Reshape((dim_out[0] // 4, dim_out[1] // 4, 64)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.UpSampling2D(interpolation='bilinear', data_format='channels_last'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Conv2DTranspose(filters=dim_out[-1], kernel_size=5, strides=1, activation=final_activation,
                                        padding='same'),
        tf.keras.layers.UpSampling2D(interpolation='bilinear', data_format='channels_last'),
        tf.keras.layers.Flatten()])


def mixture_network(dim_in, dim_out, batch_norm, name):
    net = tf.keras.Sequential(name=name)
    net.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=dim_out, activation='softmax'))
    return net


def precision_prior_params(data, num_classes, pseudo_inputs_per_class):
    # load the data into RAM to support sample with replacement
    x = []
    y = []
    for batch in data:
        x.append(batch['image'])
        y.append(batch['label'])
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)

    # git distribution of precision across pixel positions
    variance = tf.math.reduce_variance(tf.keras.layers.Flatten()(x), axis=0)
    precision = 1 / tf.clip_by_value(variance, clip_value_min=(1 / 255), clip_value_max=np.inf)
    a, _, b_inv = gamma.fit(precision, floc=0)
    b = 1 / b_inv

    # randomly select pseudo inputs
    u = []
    for i in range(num_classes):
        i_choice = np.random.choice(np.where(y == i)[0], size=pseudo_inputs_per_class, replace=False)
        u.append(tf.gather(params=x, indices=i_choice, axis=0))
    u = tf.concat(u, axis=0)

    return a, b, u


class VAE(tf.keras.Model):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, num_mc_samples):
        super(VAE, self).__init__()
        assert isinstance(dim_x, list) or isinstance(dim_x, tuple)
        assert isinstance(dim_z, int) and dim_z > 0
        assert architecture in {'dense', 'convolution'}
        assert isinstance(batch_norm, bool) and not (batch_norm and architecture == 'convolution')
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0

        # save configuration
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_mc_samples = num_mc_samples

        # set prior for latent normal
        self.pz = tfp.distributions.Normal(loc=tf.zeros(self.dim_z), scale=tf.ones(self.dim_z))
        self.pz = tfp.distributions.Independent(self.pz, reinterpreted_batch_ndims=1)

        # flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # encoder
        encoder = encoder_dense if architecture == 'dense' else encoder_convolution
        self.qz = encoder(dim_x, 2 * dim_z, batch_norm, name='qz')
        self.qz.add(tfp.layers.IndependentNormal(dim_z))

    def param_shape(self, z_samples):
        param_shape = tf.stack((tf.constant(self.num_mc_samples, dtype=tf.int32),           # MC sample dimension
                                tf.shape(z_samples)[1],                                     # batch dimension
                                tf.constant(np.prod(self.dim_x), dtype=tf.int32)), axis=0)  # event dimension
        return param_shape

    def posterior_predictive_checks(self, x):

        # sample z's variational posterior for monte-carlo estimates
        z_samples = self.qz(x).sample(sample_shape=self.num_mc_samples)

        # get posterior predictive distribution
        px_x = self.posterior_predictive(*self.z_dependent_parameters(z_samples), **self.ppc_kwargs())

        # compute first and second moments
        x_mean = tf.reshape(tf.reduce_mean(px_x.mean(), axis=0), [-1] + list(self.dim_x))
        x_std = tf.reshape(tf.reduce_mean(px_x.stddev(), axis=0), [-1] + list(self.dim_x))

        # approximately sample p(x|x)
        i_best = tf.argmax(px_x.log_prob(self.flatten(x)), axis=0)
        x_new = tf.reshape(tf.gather(px_x.sample(), i_best), [-1] + list(self.dim_x))

        return x_mean, x_std, x_new, None

    def posterior_predictive_log_prob(self, x, params, **kwargs):

        # monte-carlo estimate log posterior predictive likelihood
        lpp = self.posterior_predictive(*params, **kwargs).log_prob(x)
        return tf.reduce_logsumexp(lpp - tf.math.log(float(self.num_mc_samples)), axis=0)

    def call(self, inputs, **kwargs):
        x = inputs['image']

        # variational family q(z;x)
        qz_x = self.qz(x)

        # flatten input since here on out we treat it as a vector
        x = self.flatten(x)

        # variational objective
        elbo, ll, dkl_z, dkl_p, lppl = self.variational_objective(x, qz_x)
        self.add_loss(-tf.reduce_mean(elbo))

        # observe ELBO components
        self.add_metric(elbo, name='ELBO', aggregation='mean')
        self.add_metric(ll, name='ELL', aggregation='mean')
        self.add_metric(dkl_z, name='DKL(z)', aggregation='mean')
        if dkl_p is not None:
            self.add_metric(dkl_p, name='DKL(p)', aggregation='mean')

        # observe posterior predictive log likelihood
        self.add_metric(lppl, name='LPPL', aggregation='mean')

        return tf.constant(0.0, dtype=tf.float32)


class FixedVarianceNormalVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, variance, num_mc_samples):
        super(FixedVarianceNormalVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples)
        assert variance > 0

        # save configuration
        self.standard_deviation = tf.constant(variance ** 0.5, dtype=tf.float32)

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # decoder
        self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')

    def z_dependent_parameters(self, z_samples):

        # output parameter shape
        param_shape = self.param_shape(z_samples)

        # vectorized network calls
        z_samples = tf.reshape(z_samples, [-1, self.dim_z])
        mu = tf.reshape(self.mu(z_samples), param_shape)

        return (mu,)

    def likelihood(self, mu):
        px = tfp.distributions.Normal(loc=mu, scale=self.standard_deviation)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)

    def variational_objective(self, x, qz_x):

        # sample z's variational posterior for monte-carlo estimates
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)

        # compute parameters with dependence on z
        mu = self.z_dependent_parameters(z_samples)[0]

        # monte-carlo estimate expected log likelihood
        ell = tf.reduce_mean(self.likelihood(mu).log_prob(x), axis=0)

        # compute KL divergence w.r.t. p(z)
        dkl_z = qz_x.kl_divergence(self.pz)

        # evidence lower bound
        elbo = ell - dkl_z

        # log posterior predictive likelihood
        lpp = self.posterior_predictive_log_prob(x, params=(mu,))

        return elbo, ell, dkl_z, None, lpp

    @staticmethod
    def ppc_kwargs():
        return dict()

    def posterior_predictive(self, mu, **kwargs):
        px = tfp.distributions.Normal(loc=mu, scale=self.standard_deviation)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)


class NormalVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, split_decoder, num_mc_samples, grad_adjust='None', **kwargs):
        super(NormalVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples)
        assert isinstance(split_decoder, bool)
        assert grad_adjust in {'None', 'Normalized', 'Fixed'}
        assert isinstance(kwargs.get('a'), (type(None), float))
        assert isinstance(kwargs.get('b'), (type(None), float))

        # save configuration
        self.split_decoder = split_decoder
        self.grad_adjust = grad_adjust
        self.a = None if kwargs.get('a') is None else tf.constant(kwargs.get('a'), dtype=tf.float32)
        self.b = None if kwargs.get('b') is None else tf.constant(kwargs.get('b'), dtype=tf.float32)

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # decoder
        if self.split_decoder:
            self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
            self.log_sigma = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='sigma_x')
            self.mu_log_sigma = lambda z: (self.mu(z), self.log_sigma(z))
        else:
            dim_out = 2 * dim_out if architecture == 'dense' else list(dim_out[:-1]) + [2 * dim_out[-1]]
            self.mu_log_sigma_network = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x_sigma_x')
            self.mu_log_sigma = lambda z: tf.split(self.mu_log_sigma_network(z), num_or_size_splits=2, axis=-1)

    def z_dependent_parameters(self, z_samples):

        # output parameter shape
        param_shape = self.param_shape(z_samples)

        # vectorized network calls
        z_samples = tf.reshape(z_samples, [-1, self.dim_z])
        mu, log_sigma = self.mu_log_sigma(z_samples)
        mu = tf.reshape(mu, param_shape)
        sigma = tf.nn.softplus(tf.reshape(log_sigma, param_shape))

        return mu, sigma

    @staticmethod
    def likelihood(mu, sigma):
        px = tfp.distributions.Normal(loc=mu, scale=sigma)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)

    def variational_objective(self, x, qz_x):

        # sample z's variational posterior for monte-carlo estimates
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)

        # compute parameters with dependence on z
        mu, sigma = self.z_dependent_parameters(z_samples)

        # monte-carlo estimate expected log likelihood
        if self.grad_adjust == 'None':
            ell = tf.reduce_mean(self.likelihood(mu, sigma).log_prob(x), axis=0)
        elif self.grad_adjust == 'Normalized':
            precisions = tf.stop_gradient(sigma ** -2)
            weighted_squared_error = tf.reduce_sum(tf.math.squared_difference(x, mu) * precisions, axis=-1)
            ll1 = -weighted_squared_error / tf.reduce_sum(precisions, axis=-1) / 2
            ll2 = tfp.distributions.MultivariateNormalDiag(tf.stop_gradient(mu), scale_diag=sigma).log_prob(x)
            ell = tf.reduce_mean(ll1 + ll2, axis=0) / 2
        elif self.grad_adjust == 'Fixed':
            ll1 = tfp.distributions.MultivariateNormalDiag(mu, scale_identity_multiplier=1.).log_prob(x)
            ll2 = tfp.distributions.MultivariateNormalDiag(tf.stop_gradient(mu), scale_diag=sigma).log_prob(x)
            ell = tf.reduce_mean(ll1 + ll2, axis=0) / 2

        # compute KL divergence w.r.t. p(z)
        dkl_z = qz_x.kl_divergence(self.pz)

        # MAP-VAE option: adds log p(precision) to ELBO
        if self.b is not None:
            pp = tfp.distributions.Gamma(tf.constant(1.0, dtype=tf.float32) if self.a is None else self.a, self.b)
            ll_precision = tf.reduce_mean(tf.reduce_sum(pp.log_prob(sigma ** -2), axis=-1), axis=0)
        else:
            ll_precision = tf.constant(0.0, dtype=tf.float32)

        # evidence lower bound
        elbo = ell - dkl_z + ll_precision

        # log posterior predictive likelihood
        lpp = self.posterior_predictive_log_prob(x, params=(mu, sigma))

        return elbo, ell, dkl_z, None, lpp

    @staticmethod
    def ppc_kwargs():
        return dict()

    @staticmethod
    def posterior_predictive(mu, sigma, **kwargs):
        px = tfp.distributions.Normal(loc=mu, scale=sigma)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)


class StudentVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, min_dof, num_mc_samples=1):
        super(StudentVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples)
        assert min_dof >= 0

        # save configuration
        self.min_dof = min_dof

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # decoder
        self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
        self.nu = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='nu_x')
        self.sigma = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='sigma_x')

    def z_dependent_parameters(self, z_samples):

        # output parameter shape
        param_shape = self.param_shape(z_samples)

        # vectorized network calls
        z_samples = tf.reshape(z_samples, [-1, self.dim_z])
        mu = tf.reshape(self.mu(z_samples), param_shape)
        nu = tf.reshape(self.nu(z_samples), param_shape)
        sigma = tf.reshape(self.sigma(z_samples), param_shape)

        return mu, nu, sigma

    def likelihood(self, mu, nu, sigma):
        px = tfp.distributions.StudentT(df=nu + self.min_dof, loc=mu, scale=sigma)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)

    def variational_objective(self, x, qz_x):

        # sample z's variational posterior for monte-carlo estimates
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)

        # compute parameters with dependence on z
        mu, nu, sigma = self.z_dependent_parameters(z_samples)

        # monte-carlo estimate expected log likelihood
        ell = tf.reduce_mean(self.likelihood(mu, nu, sigma).log_prob(x), axis=0)

        # compute KL divergence w.r.t. p(z)
        dkl_z = qz_x.kl_divergence(self.pz)

        # evidence lower bound
        elbo = ell - dkl_z

        # log posterior predictive likelihood
        lpp = self.posterior_predictive_log_prob(x, params=(mu, nu, sigma))

        return elbo, ell, dkl_z, None, lpp

    @staticmethod
    def ppc_kwargs():
        return dict()

    def posterior_predictive(self, mu, nu, sigma, **kwargs):
        px = tfp.distributions.StudentT(df=nu + self.min_dof, loc=mu, scale=sigma)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)


class VariationalVarianceVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, prior_type, num_mc_samples=1, **kwargs):
        super(VariationalVarianceVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples)
        assert prior_type in {'MLE', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}

        # save configuration
        self.prior_type = prior_type

        # configure prior
        if self.prior_type == 'Standard':
            a = tf.constant([kwargs.get('a')] * np.prod(dim_x), dtype=tf.float32)
            b = tf.constant([kwargs.get('b')] * np.prod(dim_x), dtype=tf.float32)
            self.pp = self.gamma(a, b)
        elif 'VAMP' in self.prior_type:
            # pseudo-inputs
            trainable = '*' in self.prior_type
            self.u = tf.Variable(initial_value=kwargs.get('u'), dtype=tf.float32, trainable=trainable, name='u')
        elif self.prior_type == 'VBEM':
            # fixed prior parameters for precision
            params = [0.05, 0.1, 0.25, 0.5, 0.75, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            uv = softplus_inverse(np.array(tuple(itertools.product(params, params)), dtype=np.float32).T)
            u = tf.expand_dims(uv[0], axis=-1)
            v = tf.expand_dims(uv[1], axis=-1)
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=False, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=False, name='v')
        elif self.prior_type == 'VBEM*':
            # trainable prior parameters for precision
            k = kwargs.get('k')
            u = tf.random.uniform(shape=(k, np.prod(dim_x)), minval=-3, maxval=3, dtype=tf.float32)
            v = tf.random.uniform(shape=(k, np.prod(dim_x)), minval=-3, maxval=3, dtype=tf.float32)
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # build parameter networks
        self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
        self.alpha = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='alpha_x')
        self.beta = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='beta_x')
        if self.prior_type in {'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*'}:
            self.pi = mixture_network(dim_z, self.u.shape[0], batch_norm, name='pi')
            self.pc = tfp.distributions.Categorical(logits=[1] * self.u.shape[0])

    def z_dependent_parameters(self, z_samples):

        # output parameter shape
        param_shape = self.param_shape(z_samples)

        # vectorized network calls
        z_samples = tf.reshape(z_samples, [-1, self.dim_z])
        mu = tf.reshape(self.mu(z_samples), param_shape)
        alpha = tf.reshape(self.alpha(z_samples), param_shape)
        beta = tf.reshape(self.beta(z_samples), param_shape)

        return mu, alpha, beta

    @staticmethod
    def gamma(alpha, beta):
        prior = tfp.distributions.Gamma(alpha, beta)
        return tfp.distributions.Independent(prior, reinterpreted_batch_ndims=1)

    @staticmethod
    def qp(alpha, beta):
        return tfp.distributions.Independent(tfp.distributions.Gamma(alpha, beta))

    def dkl_precision(self, z_samples, p_samples, alpha, beta, vamp_samples=None):

        # variational family q(precision|z)
        qp = tfp.distributions.Independent(tfp.distributions.Gamma(alpha, beta))

        # compute kl-divergence depending on prior type
        if self.prior_type == 'Standard':
            dkl = qp.kl_divergence(self.pp)
        elif 'VAMP' in self.prior_type or 'VBEM' in self.prior_type:

            # compute prior's mixture proportions
            if self.prior_type in {'VAMP', 'VAMP*'}:
                pi = self.pi(z_samples)
            else:
                pi = tf.ones(self.u.shape[0]) / self.u.shape[0]

            # compute prior's mixture components
            if 'VAMP' in self.prior_type:
                alpha = self.alpha(self.qz(self.u).sample())
                beta = self.beta(self.qz(self.u).sample())
            else:
                alpha = tf.nn.softplus(self.u)
                beta = tf.nn.softplus(self.v)
            pp_c = self.gamma(alpha + EPSILON, beta + EPSILON)

            # MC estimate kl-divergence due to pesky log-sum
            p_samples = tf.clip_by_value(p_samples, clip_value_min=EPSILON, clip_value_max=tf.float32.max)
            p_samples = tf.tile(tf.expand_dims(p_samples, axis=-2), [1, 1] + pp_c.batch_shape.as_list() + [1])
            log_pi = tf.math.log(tf.expand_dims(pi, axis=0))
            log_pp_c = tf.clip_by_value(pp_c.log_prob(p_samples), clip_value_min=tf.float32.min, clip_value_max=100)
            log_pp = tf.reduce_logsumexp(log_pi + log_pp_c, axis=-1)
            dkl = -qp.entropy() - tf.reduce_mean(log_pp, axis=0)

        else:
            dkl = tf.constant(0.0, dtype=tf.float32)

        return dkl

    def variational_objective(self, x, qz_x):

        # sample z's variational posterior for monte-carlo estimates
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)

        # compute parameters with dependence on z
        mu, alpha, beta = self.z_dependent_parameters(z_samples)

        # monte-carlo estimate expected log likelihood
        ell = tf.reduce_mean(expected_log_normal(x, mu, alpha, beta), axis=0)

        # compute KL divergence w.r.t. p(z)
        dkl_z = qz_x.kl_divergence(self.pz)

        # sample p's variational posterior for monte-carlo estimates
        p_samples = self.gamma(alpha, beta).sample()

        # monte-carlo estimate KL divergence w.r.t. p(lambda)
        # dkl_p = tf.vectorized_map(lambda elements: self.dkl_precision(*elements), (z_samples, p_samples, alpha, beta))
        # dkl_p = tf.reduce_mean(dkl_p, axis=0)
        dkl_p = tf.reduce_mean(self.gamma(alpha, beta).kl_divergence(self.pp), axis=0)

        # evidence lower bound
        elbo = ell - dkl_z - dkl_p

        # log posterior predictive likelihood
        lpp = self.posterior_predictive_log_prob(x, params=(mu, alpha, beta), **{'analytic_integration': True})

        return elbo, ell, dkl_z, dkl_p, lpp

    @staticmethod
    def ppc_kwargs():
        return {'analytic_integration': False}

    def posterior_predictive(self, mu, alpha, beta, **kwargs):
        assert isinstance(kwargs.get('analytic_integration'), bool)
        if kwargs.get('analytic_integration'):
            px = tfp.distributions.StudentT(df=2 * alpha, loc=mu, scale=tf.sqrt(beta / alpha))
        else:
            p_samples = self.gamma(alpha, beta).sample()
            px = tfp.distributions.Normal(loc=mu, scale=p_samples ** -0.5)
        return tfp.distributions.Independent(px, reinterpreted_batch_ndims=1)


if __name__ == '__main__':
    # set configuration
    PX_FAMILY = 'Normal'
    BATCH_SIZE = 250
    ARCH = 'dense'
    BATCH_NORM = False
    DIM_Z = 10
    NUM_MC_SAMPLES = 10
    PSEUDO_INPUTS_PER_CLASS = 10

    # load the data set
    train_set, test_set, info = load_data_set(data_set_name='mnist', px_family=PX_FAMILY, batch_size=BATCH_SIZE)
    DIM_X = info.features['image'].shape

    # get precision prior parameters
    A, B, U = precision_prior_params(data=train_set,
                                     num_classes=info.features['label'].num_classes,
                                     pseudo_inputs_per_class=PSEUDO_INPUTS_PER_CLASS)

    # # VAE with fixed decoder variance
    # vae = FixedVarianceNormalVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                              num_mc_samples=NUM_MC_SAMPLES, variance=1)

    # # VAE with shared mean/variance decoder network
    # vae = NormalVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                 num_mc_samples=NUM_MC_SAMPLES, split_decoder=False)

    # # VAE with split mean/variance decoder network
    # vae = NormalVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                 num_mc_samples=NUM_MC_SAMPLES, split_decoder=True)

    # # MAP VAE (Takahashi, 2018)
    # vae = NormalVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                 num_mc_samples=NUM_MC_SAMPLES, split_decoder=True, b=1e-3)

    # # Student-T VAE (Takahashi, 2018)
    # vae = StudentVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                  num_mc_samples=NUM_MC_SAMPLES, min_dof=3)

    # Empirical-Bayes MAP VAE (ours) # TODO: is this worth trying again?
    # vae = NormalVAE(dim_x=DIM_X, architecture=ARCH, batch_norm=BATCH_NORM, split_decoder=True, a=A, b=B)

    # Variational Variance VAE (ours) + standard prior
    vae = VariationalVarianceVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
                                 num_mc_samples=NUM_MC_SAMPLES, prior_type='Standard', a=1., b=1e-3)

    # # Variational Variance VAE (ours) + VBEM prior
    # vae = VariationalVarianceVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                              prior_type='VBEM', k=10)

    # Variational Variance VAE (ours) + VAMP prior
    # vae = VariationalVarianceVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
    #                              prior='vamp', a=1., b=1e-3, u=U)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=[None], run_eagerly=False)

    # train the model
    vae.fit(train_set, validation_data=test_set, epochs=1000, verbose=0,
            validation_steps=np.ceil(info.splits['test'].num_examples // BATCH_SIZE),
            callbacks=[LearningCurveCallback(train_set),
                       ReconstructionCallback(train_set, info.features['label'].num_classes),
                       LatentVisualizationCallback2D(vae.dim_x, vae.dim_z),
                       tf.keras.callbacks.EarlyStopping(monitor='val_LPPL', min_delta=1.0, patience=50, mode='max')])
    print('Done!')

    # keep plots open
    plt.show()
