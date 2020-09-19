import time
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

VAE_METRICS = {
    'ELBO':         'ELBO',
    'ELL':          'Expected Log Likelihood',
    'DKL(z)':       '$D_{KL}(q(z|x)||p(z))$',
    'DKL(p)':       '$D_{KL}(q(\\lambda|z)||p(\\lambda))$',
    'LPPL':         'Log Posterior Predictive Likelihood',
    'RMSE(mean)':   'RMSE(Posterior Predictive Mean)',
    'RMSE(sample)': 'RMSE(Posterior Predictive Sample)'
}


class RegressionCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs, parallel=False):
        super().__init__()
        self.n_epochs = n_epochs
        self.parallel = parallel

    def on_epoch_end(self, epoch, logs=None):
        if not self.parallel and epoch % 500 == 0:
            prefix = 'val_' if sum(['val_' in key for key in logs.keys()]) else ''
            print('Epoch {:d}/{:d},'.format(epoch, self.n_epochs),
                  'Loss {:.4f},'.format(logs[prefix + 'loss']),
                  'LL {:.4f},'.format(logs[prefix + 'LL']),
                  'KL {:.4f}'.format(logs[prefix + 'KL']),
                  'LL Adjusted {:.4f},'.format(logs[prefix + 'LL (adjusted)']),
                  'RMSE {:.4f}'.format(np.sqrt(logs[prefix + 'MSE'])))


class LearningCurveCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_set):
        super().__init__()
        self.train_set = train_set
        self.history = dict()
        self.fig = plt.figure(figsize=(12, 4))
        self.start_time = None

    def __update_history(self, logs):
        for key, val in logs.items():
            if 'loss' in key:
                val = np.mean(val)
            if key not in self.history.keys():
                self.history.update({key: [val]})
            else:
                self.history[key].append(val)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # update metric history
        self.__update_history(logs)
        if epoch == 0:
            return

        # clear the figure
        self.fig.clear()

        # compute number of rows
        num_rows = 1
        row = 0

        # plot VI terms
        row += 1
        vi_keys = [key for key in logs.keys() if key in VAE_METRICS.keys() and 'val_' not in key]
        for i, key in enumerate(vi_keys):
            sp = self.fig.add_subplot(num_rows, len(vi_keys), (row - 1) * len(vi_keys) + i + 1)
            sp.set_title(VAE_METRICS[key], fontsize=9)
            sp.plot(self.history[key][1:], label='train')
            sp.plot(self.history['val_' + key][1:], label='test')
            sp.legend()

        plt.pause(0.05)

        # print update
        loss = self.history['loss'][-1]
        duration = time.time() - self.start_time
        print('Epoch {}: loss = {:.2f}, duration = {:f}'.format(epoch, loss, duration))


class ReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_set, num_classes, chain_len=1):
        super().__init__()
        assert isinstance(chain_len, int) and chain_len >= 1
        self.train_set = train_set
        self.num_classes = num_classes
        self.chain_len = chain_len
        self.fig = plt.figure()
        self.fig_size_set = False

    @ staticmethod
    def __add_color_bar(sp, ax, vmin, vmax):
        cax = inset_axes(sp,
                         width='10%',
                         height='80%',
                         bbox_transform=sp.transAxes,
                         bbox_to_anchor=(0.25, 0, 1.0, 1.0),
                         loc='right')
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(cax, cmap=ax.cmap, norm=norm)

    def on_epoch_end(self, epoch, logs=None):

        # initialize data containers
        class_done = np.zeros(self.num_classes, dtype=bool)
        x_orig = np.zeros([self.num_classes, 1] + list(self.model.dim_x), dtype=np.float32)
        z_mean = np.zeros((self.num_classes, self.chain_len, self.model.dim_z))
        z_std = np.zeros((self.num_classes, self.chain_len, self.model.dim_z))
        x_mean = np.zeros([self.num_classes, self.chain_len] + list(self.model.dim_x))
        x_std = np.zeros([self.num_classes, self.chain_len] + list(self.model.dim_x))
        x_new = np.zeros([self.num_classes, self.chain_len] + list(self.model.dim_x))

        # loop until we have an example for each class
        batch_iterator = iter(self.train_set)
        while sum(class_done) < self.num_classes:
            batch = next(batch_iterator)
            for k in range(self.num_classes):
                if not class_done[k]:
                    i_match = np.where(batch['label'] == k)[0]
                    if len(i_match):
                        class_done[k] = True
                        i = np.random.choice(i_match)
                        x_orig[k] = batch['image'][i:i + 1]
                        x = x_orig[k]
                        for c in range(self.chain_len):
                            qz_x = self.model.qz(x)
                            z_mean[k, c] = np.squeeze(qz_x.mean())
                            z_std[k, c] = np.squeeze(qz_x.stddev())
                            x_mean[k, c], x_std[k, c], x_new[k, c] = self.model.posterior_predictive_checks(x=x)
                            x = x_new[k, c]

        # plot results
        self.fig.clear()
        assert self.model.dim_x[-1] in {1, 3}
        num_rows = 1 + 4 * self.chain_len
        for i in range(self.num_classes):

            # initialize offset
            offset = 1

            # generate subplots for original data
            sp = self.fig.add_subplot(num_rows, self.num_classes, i + offset)
            ax = sp.imshow(np.squeeze(x_orig[i]), vmin=0, vmax=1)
            sp.set_xticks([])
            sp.set_yticks([])
            if i == 0:
                sp.set_ylabel('Original')
            if i == self.num_classes - 1:
                self.__add_color_bar(sp=sp, ax=ax, vmin=0, vmax=1)

            # loop over the chain
            for c in range(self.chain_len):

                # generate subplots for z's variational parameters
                offset += self.num_classes
                sp = self.fig.add_subplot(num_rows, self.num_classes, i + offset)
                sp.errorbar(np.arange(self.model.dim_z), z_mean[i, c], yerr=2 * z_std[i, c])
                sp.set_xticks([])
                sp.yaxis.tick_right()
                sp.set_ylim([np.min(z_mean - z_std), np.max(z_mean + z_std)])
                if i == 0:
                    sp.set_ylabel('$q(z|x_i)$')
                if i < self.num_classes - 1:
                    sp.set_yticks([])

                # generate subplots for mean(x) under the likelihood model
                offset += self.num_classes
                sp = self.fig.add_subplot(num_rows, self.num_classes, i + offset)
                ax = sp.imshow(np.squeeze(x_mean[i][c]))
                sp.set_xticks([])
                sp.set_yticks([])
                if i == 0:
                    sp.set_ylabel('mean$(x_i)$')
                if i == self.num_classes - 1:
                    self.__add_color_bar(sp=sp, ax=ax, vmin=np.min(x_mean[i]), vmax=np.max(x_mean[i]))

                # generate subplots for std(x) under the likelihood model
                offset += self.num_classes
                sp = self.fig.add_subplot(num_rows, self.num_classes, i + offset)
                x_std_plot = x_std[i][c] if x_std[i][c].shape[-1] == 1 else tf.image.rgb_to_grayscale(x_std[i][c]).numpy()
                ax = sp.imshow(np.squeeze(x_std_plot), vmin=np.min(x_std[i]), vmax=np.max(x_std[i]))
                sp.set_xticks([])
                sp.set_yticks([])
                if i == 0:
                    sp.set_ylabel('std$(x_i)$')
                if i == self.num_classes - 1:
                    self.__add_color_bar(sp=sp, ax=ax, vmin=np.min(x_std[i]), vmax=np.max(x_std[i]))

                # generate subplots for x ~ p(x'|x) (the variational posterior predictive)
                offset += self.num_classes
                sp = self.fig.add_subplot(num_rows, self.num_classes, i + offset)
                ax = sp.imshow(np.squeeze(x_new[i][c]))
                sp.set_xticks([])
                sp.set_yticks([])
                if i == 0:
                    sp.set_ylabel('x\' ~ p(x\'|x)')
                if i == self.num_classes - 1:
                    self.__add_color_bar(sp=sp, ax=ax, vmin=np.min(x_new[i]), vmax=np.max(x_new[i]))

        # format figures
        if not self.fig_size_set:
            self.fig_size_set = True
            self.fig.set_size_inches(self.num_classes, num_rows)
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.93, top=1, wspace=0.05, hspace=0.05)


class LatentVisualizationCallback2D(tf.keras.callbacks.Callback):

    def __init__(self, dim_x, dim_z):
        super().__init__()
        assert isinstance(dim_x, list) or isinstance(dim_x, tuple)
        assert isinstance(dim_z, int) and dim_z > 0

        # make figure only if dimensions are supported
        if dim_z == 2 and (dim_x[-1] == 1 or dim_x[-1] == 3):
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
            self.ax = np.reshape(self.ax, -1)

    def __plot_latent_representation_2_dims(self):

        # generate the latent encoding test points
        extreme = 3.0  # corresponds to the number of standard deviations away from the prior's mean
        num = 15
        lp = np.linspace(-extreme, extreme, num)
        zx, zy = np.meshgrid(lp, lp, indexing='xy')
        z = np.zeros([0, 2])
        for x in range(num):
            for y in range(num):
                z_new = np.array([[zx[x, y], zy[x, y]]])
                z = np.concatenate((z, z_new), axis=0)

        # generate reconstruction
        x_latent = self.model.posterior_predictive_checks(z=np.float32(z))[0]

        # loop over the channels
        x_plot = []
        for c in range(self.model.dim_x[-1]):

            # grab all reconstructions for this class/channel
            x_kc = x_latent[0:num ** 2, :, :, c]

            # turn them into a block
            x_block = []
            for y in range(num):
                x_row = []
                for x in range(num):
                    x_row.append(x_kc[y * num + x])
                x_block.insert(0, x_row)
            x_plot.append(np.block(x_block))

        # generate subplots for original data
        sp = self.ax[0]
        sp.cla()
        sp.imshow(np.squeeze(np.stack(x_plot, axis=2)), vmin=0, vmax=1)
        sp.set_xticks([])
        sp.set_yticks([])

    def on_epoch_end(self, epoch, logs=None):
        if self.model.dim_z == 2:
            self.__plot_latent_representation_2_dims()
