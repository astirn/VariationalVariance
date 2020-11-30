#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:25:43 2019

@author: nsde
"""

#%%
import argparse
import numpy as np
import torch
from torch import nn
# from torchvision.utils import save_image
from utils import get_image_dataset
from torch import distributions as D
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from utils import batchify, dist, translatedSigmoid, RBF2, PosLinear, Reciprocal, logmeanexp, t_likelihood
from itertools import chain
from locality_sampler import gen_Qw, locality_sampler, get_pseupoch, local_batchify
from sklearn.cluster import KMeans
sns.set()

#%%
class BatchFlatten(nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x.reshape(n, -1)

#%%
class BatchReshape(nn.Module):
    def __init__(self, *s):
        super(BatchReshape, self).__init__()
        self.s = s
        
    def forward(self, x):
        n = x.shape[0]
        return x.reshape(n, *self.s)

#%%
def argparser():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    gs = parser.add_argument_group('General settings')
    gs.add_argument('--model', type=str, default='vae', help='model to use')
    gs.add_argument('--dataset', type=str, default='fashionmnist', help='dataset to use')
    gs.add_argument('--cuda', type=bool, default=True, help='use cuda')
    
    ms = parser.add_argument_group('Model specific settings')
    ms.add_argument('--batch_size', type=int, default=512, help='batch size')
    ms.add_argument('--shuffel', type=bool, default=True, help='shuffel data during training')
    ms.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ms.add_argument('--beta', type=float, default=1.0, help='scaling of kl term')
    ms.add_argument('--iters', type=int, default=100, help='number of iterations')
    ms.add_argument('--latent_size', type=int, default=10, help='latent space size')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
class basemodel(nn.Module):
    def __init__(self, in_size, direc, latent_size=2, cuda=True):
        super(basemodel, self).__init__()
        self.switch = 0.0
        self.direc = direc
        self.c = in_size[0]
        self.h = in_size[1]
        self.w = in_size[2]
        self.in_size = np.prod(in_size)
        self.latent_size = latent_size
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        
    def encoder(self, x):
        # return self.enc_mu(x), self.enc_var(x)
        mu_var = self.enc(x)
        new_shape = list(mu_var.shape)
        new_shape[-1] = self.latent_size
        mu_var = mu_var.reshape([-1, 2 * self.latent_size])
        mu = mu_var[:, :self.latent_size].reshape(new_shape)
        var = nn.Softplus()(mu_var[:, self.latent_size:].reshape(new_shape))
        return mu, var

    def decoder(self, z):
        x_mu, x_var = self.dec_mu(z), self.dec_var(z)
        x_var = self.switch * x_var + (1-self.switch)*torch.tensor([0.02**2], device=z.device)
        return x_mu, x_var

    def sample(self, N):
        z = torch.randn(N, self.latent_size, device=self.device)
        x_mu, x_var = self.decoder(z)
        return x_mu, x_var
    
    def forward(self, x, beta=1.0, epsilon=1e-2):
        
        z_mu, z_var = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, torch.sqrt(z_var+epsilon)), 1)
        z = q_dist.rsample()
        x_mu, x_var = self.decoder(z)
        if self.switch:
            p_dist = D.Independent(D.Normal(x_mu, torch.sqrt(x_var+epsilon)), 1)
        else:
            p_dist = D.Independent(D.Normal(x_mu, 1.0), 1)
            # p_dist = D.Independent(D.Bernoulli(x_mu), 1)
        
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = log_px - beta*kl
        return elbo.mean(), log_px, kl, x_mu, x_var, z, z_mu, z_var
    
    def evaluate(self, X, L=10):
        with torch.no_grad():
            x_mu, x_var = self.sample(L)
            parzen_dist = D.Independent(D.Normal(x_mu, x_var), 1)
            elbolist, logpxlist, parzen_score = [ ], [ ], [ ]
            for x in tqdm(X, desc='evaluating', unit='samples'):
                x = torch.tensor(x.reshape(1, -1), device=self.device)
                elbo, logpx, _, _, _, _, _, _ = self.forward(x)
                elbolist.append(elbo.item())
                logpxlist.append(logpx.mean().item())
                score = parzen_dist.log_prob(x) # unstable sometimes
                parzen_score.append(torch.logsumexp(score[torch.isfinite(score)],dim=0).item())
            
            return np.array(elbolist), np.array(logpxlist), np.array(parzen_score)
    
    def save_something(self, name, data):
        pass
        # current_state = self.training
        # self.eval()
        # 
        # x = torch.tensor(data).to(self.device)
        # 
        # # Save reconstructions
        # _, _, _, x_mu, x_var, z, z_mu, z_var = self.forward(x)
        # 
        # temp1 = x[:10].reshape(-1, self.c, self.h, self.w)
        # temp2 = x_mu[:10].clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        # temp3 = torch.normal(x_mu[:10], x_var[:10]).clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        # 
        # save_image(torch.cat([temp1, temp2, temp3], dim=0), 
        #            self.direc + '/' + name + '_recon.png', nrow=10)
        # 
        # # Make grid from latent space
        # if self.latent_size == 2:
        #     size = 50
        #     grid = np.stack([m.flatten() for m in np.meshgrid(np.linspace(-4,4,size), np.linspace(4,-4,size))]).T.astype('float32')
        #     x_mu, x_var = model.decoder(torch.tensor(grid).to(model.device))
        #     temp1 = x_mu.clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        #     temp2 = torch.normal(x_mu, x_var).clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        #     
        #     save_image(temp1, self.direc + '/' + name + '_grid1.png',
        #                nrow=size)
        #     
        #     save_image(temp2, self.direc + '/' + name + '_grid2.png',
        #                nrow=size)
        #     
        #     plt.figure()
        #     plt.imshow(x_var.sum(dim=1).log().reshape(size,size).detach().cpu().numpy())
        #     plt.colorbar()
        #     plt.savefig(self.direc + '/' + name + '_variance.png')
        #     
        #     
        # # Make plot of latent points
        # if self.latent_size == 2:
        #     plt.figure()
        #     plt.plot(z[:,0].detach().cpu().numpy(), z[:,1].detach().cpu().numpy(),'.')
        #     if hasattr(self, "C"):
        #         plt.plot(self.C[:,0].detach().cpu().numpy(), self.C[:,1].detach().cpu().numpy(),'.')
        #     plt.savefig(direc + '/' + name + '_latents.png')
        #     
        # # Make samples
        # x_mu, x_var = self.sample(100)
        # temp1 = x_mu.clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        # temp2 = torch.normal(x_mu, x_var).clamp(0.0,1.0).reshape(-1, self.c, self.h, self.w)
        # 
        # save_image(temp1, self.direc + '/' + name + '_samples1.png', nrow=10)
        # 
        # save_image(temp2, self.direc + '/' + name + '_samples2.png', nrow=10)
        # 
        # self.training = current_state
    
#%%
class vae(basemodel):
    def __init__(self, in_size, direc, latent_size=2, cuda=True):
        super(vae, self).__init__(in_size, direc, latent_size, cuda)
        # commented out to make sure we only need the declarations in john class
        # self.enc_mu = nn.Sequential(nn.Linear(self.in_size, 512),
        #                             nn.BatchNorm1d(512),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(512, 256),
        #                             nn.BatchNorm1d(256),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(256, self.latent_size))
        # self.enc_var = nn.Sequential(nn.Linear(self.in_size, 512),
        #                              nn.BatchNorm1d(512),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(512, 256),
        #                              nn.BatchNorm1d(256),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(256, self.latent_size),
        #                              nn.Softplus())
        # self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, 256),
        #                             nn.BatchNorm1d(256),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(256, 512),
        #                             nn.BatchNorm1d(512),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(512, self.in_size),
        #                             nn.Sigmoid())
        # self.dec_var = nn.Sequential(nn.Linear(self.latent_size, 256),
        #                              nn.BatchNorm1d(256),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(256, 512),
        #                              nn.BatchNorm1d(512),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(512, self.in_size),
        #                              nn.Softplus())
    
    def fit(self, Xtrain, n_iters=100, lr=1e-3, batch_size=256, beta=1.0):
        self.train()
        if self.device == torch.device('cuda'):
            self.cuda()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(chain(self.dec_var.parameters()), lr=lr)
        
        it = 0
        batches = batchify(Xtrain, batch_size = batch_size, shuffel=True)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        loss, var = [[ ],[ ],[ ]], [ ]
        while it < n_iters:
            self.switch = 1.0 if it > n_iters/2 else 0.0
            anneling = np.minimum(1, it/(n_iters/2))*beta
            x = torch.tensor(next(batches)[0], device=self.device)
            elbo, log_px, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
            
            if self.switch:
                optimizer2.zero_grad()
                (-elbo).backward()
                optimizer2.step()
            else:
                optimizer.zero_grad()
                (-elbo).backward()
                optimizer.step()
            
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item(), 'z_var': z_var.mean().item(), 'anneling': anneling})
            loss[0].append((-elbo).item())
            loss[1].append(log_px.mean().item())
            loss[2].append(kl.mean().item())
            var.append(x_var.mean().item())
            it+=1
            
            if it%2500==0:
                self.save_something('it'+str(it), Xtrain[::20])
        progressBar.close()
        return loss, var

#%%
class john(basemodel):
    def __init__(self, in_size, direc, latent_size=2, cuda=True, fixed_var=10):
        super(john, self).__init__(in_size, direc, latent_size, cuda)
        self.opt_switch = 1
        self.fixed_var = fixed_var
        # self.enc_mu = nn.Sequential(nn.Linear(self.in_size, 512),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(512, 256),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(256, self.latent_size))
        # self.enc_var = nn.Sequential(nn.Linear(self.in_size, 512),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(512, 256),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(256, self.latent_size),
        #                              nn.Softplus())
        self.enc = nn.Sequential(nn.Linear(self.in_size, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 2 * self.latent_size))

        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, 128),
                                    nn.ELU(),
                                    nn.Linear(128, 256),
                                    nn.ELU(),
                                    nn.Linear(256, 512),
                                    nn.ELU(),
                                    nn.Linear(512, self.in_size))
                                    # nn.Sigmoid())
        self.alpha = nn.Sequential(nn.Linear(self.latent_size, 128),
                                   nn.ELU(),
                                   nn.Linear(128, 256),
                                   nn.ELU(),
                                   nn.Linear(256, 512),
                                   nn.ELU(),
                                   nn.Linear(512, self.in_size),
                                   nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(self.latent_size, 128),
                                  nn.ELU(),
                                  nn.Linear(128, 256),
                                  nn.ELU(),
                                  nn.Linear(256, 512),
                                  nn.ELU(),
                                  nn.Linear(512, self.in_size),
                                  nn.Softplus())
      
    def decoder(self, z):
        x_mu = self.dec_mu(z)
        if self.switch:
            d = dist(z, self.C)
            d_min = d.min(dim=1, keepdim=True)[0]
            s = translatedSigmoid(d_min, -6.907*0.3, 0.3)
            alpha = self.alpha(z)
            beta = self.beta(z)
            gamma_dist = D.Gamma(alpha+1e-6, beta+1e-6)
            samples_var = gamma_dist.rsample([20])
            x_var = (1.0/(samples_var+1e-6))
            x_var = (1-s) * x_var + s*(self.fixed_var*torch.ones_like(x_var))
        else:
            x_var = (0.02**2)*torch.ones_like(x_mu)
            
        return x_mu, x_var        
    
    def fit(self, Xtrain, x_test, Xplot, n_iters=100, lr=1e-3, batch_size=250, n_clusters=50, beta=1.0, its_per_epoch=2500):
        self.train()
        if self.device == torch.device('cuda'):
            self.cuda()
        
        optimizer1 = torch.optim.Adam(chain(#self.enc_mu.parameters(),
                                            #self.enc_var.parameters(),
                                            self.enc.parameters(),
                                            self.dec_mu.parameters()),
                                      lr=lr)
        optimizer2 = torch.optim.Adam(chain(#self.enc_mu.parameters(),
                                            #self.enc_var.parameters(),
                                            self.enc.parameters(),
                                            self.dec_mu.parameters()),
                                      lr=lr)
        optimizer3 = torch.optim.Adam(chain(self.alpha.parameters(),
                                            self.beta.parameters()),
                                      lr=lr)
            
        it = 0
        batches = batchify(Xtrain, batch_size = batch_size, shuffel=True)
        local_batches = local_batchify(Xtrain)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        x_plot = torch.tensor(Xplot).to(torch.float32).to(self.device)
        ll_best = -np.inf
        epoch_best = np.inf
        while it < n_iters:
            self.switch = 1.0 if it > n_iters/2 else 0.0
            anneling = np.minimum(1, it/(n_iters/2))*beta
            #self.opt_switch = (self.opt_switch+1) if (it % 11 == 0 and self.switch) else self.opt_switch
            # if self.switch and (it % 1000 == 0 or not hasattr(self, "C")):
            if self.switch and not hasattr(self, "C"):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(self.encoder(torch.tensor(Xtrain).to(self.device))[0].detach().cpu().numpy())
                self.C = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
                        
            if not self.switch:
                x = next(batches)    
                x = torch.tensor(x).to(torch.float32).to(self.device)
                
                optimizer1.zero_grad()
                elbo, log_px, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
                (-elbo).backward()
                optimizer1.step()
            else:
                x, mean_w, var_w = next(local_batches)
                x = torch.tensor(x).to(torch.float32).to(self.device)
                mean_w = torch.tensor(mean_w).to(torch.float32).to(self.device)
                var_w = torch.tensor(var_w).to(torch.float32).to(self.device)
                
                elbo, logpx, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
                if self.opt_switch % 2 == 0:
                    optimizer2.zero_grad()
                    elbo = t_likelihood(x, x_mu, x_var, mean_w) - kl.mean()
                    (-elbo).backward()
                    optimizer2.step()
                else:
                    optimizer3.zero_grad()
                    elbo = t_likelihood(x, x_mu, x_var, var_w) - kl.mean()
                    (-elbo).backward()
                    optimizer3.step()
                
                elbo, log_px, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
                
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item(), 'x_var': x_var.mean().item(), 'anneling': anneling})

            # epoch complete and in second phase of training (i.e. fitting variance)
            if it % its_per_epoch == 0 and self.switch:
                self.eval()
                with torch.no_grad():

                    # initialize containers
                    ll = []
                    mean_residuals = []
                    var_residuals = []
                    sample_residuals = []

                    # loop over batches
                    for i in range(int(np.ceil(x_test.shape[0] / batch_size))):

                        # run Detlefsen network
                        i_start = i * batch_size
                        i_end = min((i + 1) * batch_size, x_test.shape[0])
                        x = torch.tensor(x_test[i_start:i_end]).to(torch.float32).to(self.device)
                        _, _, _, mu_x, sigma2_x, _, _, _ = self.forward(x, anneling)
                        mean = mu_x.cpu().numpy()
                        variance = sigma2_x.cpu().numpy()

                        # create p(x|x): a uniform mixture of Normals over the variance samples
                        components = []
                        for v in tf.unstack(variance):
                            normal = tfp.distributions.Normal(loc=mean, scale=v ** 0.5)
                            components.append(tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1))
                        cat = tfp.distributions.Categorical(logits=tf.ones((variance.shape[1], variance.shape[0])))
                        px_x = tfp.distributions.Mixture(cat=cat, components=components)

                        # append results
                        x = x.cpu().numpy()
                        ll.append(px_x.log_prob(x))
                        mean_residuals.append(px_x.mean() - x)
                        var_residuals.append(px_x.variance() - mean_residuals[-1] ** 2)
                        sample_residuals.append(px_x.sample() - x)

                    # if mean likelihood is new best
                    ll = tf.reduce_mean(tf.concat(ll, axis=0)).numpy()
                    if ll > ll_best and it > n_iters * 0.6:

                        # record best ll
                        ll_best = ll

                        # compute metrics
                        metrics = {'LL': ll_best,
                                   'Best Epoch': it // its_per_epoch,
                                   'Mean Bias': tf.reduce_mean(tf.concat(mean_residuals, axis=0)).numpy(),
                                   'Mean RMSE': tf.sqrt(tf.reduce_mean(tf.concat(mean_residuals, axis=0) ** 2)).numpy(),
                                   'Var Bias': tf.reduce_mean(tf.concat(var_residuals, axis=0)).numpy(),
                                   'Var RMSE': tf.sqrt(tf.reduce_mean(tf.concat(var_residuals, axis=0) ** 2)).numpy(),
                                   'Sample Bias': tf.reduce_mean(tf.concat(sample_residuals, axis=0)).numpy(),
                                   'Sample RMSE': tf.sqrt(tf.reduce_mean(tf.concat(sample_residuals, axis=0) ** 2)).numpy()}

                        # get p(x|x) for the held-out plotting data
                        _, _, _, mu_x, sigma2_x, _, _, _ = self.forward(x_plot, anneling)
                        mean = mu_x.cpu().numpy()
                        variance = sigma2_x.cpu().numpy()
                        components = []
                        for v in tf.unstack(variance):
                            normal = tfp.distributions.Normal(loc=mean, scale=v ** 0.5)
                            components.append(tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1))
                        cat = tfp.distributions.Categorical(logits=tf.ones((variance.shape[1], variance.shape[0])))
                        px_x = tfp.distributions.Mixture(cat=cat, components=components)

                        # save first two moments and samples for the plotting data
                        reconstruction = {'mean': px_x.mean().numpy(),
                                          'std': px_x.stddev().numpy(),
                                          'sample': px_x.sample().numpy()}

                    # early stop check
                    elif self.switch and it // its_per_epoch > epoch_best + 50:
                        print('Early Stop!')
                        break
                self.train()
            it += 1
          
        progressBar.close()
        return metrics, reconstruction
    
#%%


def detlefsen_vae_baseline(x_train, x_test, x_plot, dim_z, epochs, batch_size, fixed_var):
    orig_shape = list(x_train.shape)
    mdl = john(in_size=x_train.shape[1:], direc=None, latent_size=dim_z, cuda=True, fixed_var=fixed_var)
    x_train = np.reshape(x_train, [x_train.shape[0], -1])
    x_test = np.reshape(x_test, [x_test.shape[0], -1])
    x_plot = np.reshape(x_plot, [x_plot.shape[0], -1])
    its_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
    iterations = its_per_epoch * epochs
    metrics, reconstruction = mdl.fit(Xtrain=x_train, x_test=x_test, Xplot=x_plot,
                                      n_iters=iterations, lr=1e-4, batch_size=batch_size, its_per_epoch=its_per_epoch)
    reconstruction['mean'] = np.reshape(reconstruction['mean'], [-1] + orig_shape[1:])
    reconstruction['std'] = np.reshape(reconstruction['std'], [-1] + orig_shape[1:])
    reconstruction['sample'] = np.reshape(reconstruction['sample'], [-1] + orig_shape[1:])
    return metrics, reconstruction


if __name__ == '__main__':
    args = argparser()
    direc = 'results/vae_results/' + args.model + '_' + args.dataset
    if not direc in os.listdir():
        os.makedirs(direc, exist_ok=True)
        
    # Get main set and secondary set
    Xtrain, ytrain, Xtest, ytest = get_image_dataset(args.dataset)
    in_size = Xtrain.shape[1:]
    
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1) / 255.0
    Xtest = Xtest.reshape(Xtest.shape[0], -1) / 255.0
    
    
    # Initialize model
    if args.model == 'vae':
        model = vae(in_size, direc, args.latent_size, args.cuda)
    elif args.model == 'john':
        model = john(in_size, direc, args.latent_size, args.cuda)
    
    
    # Fitting
    loss, var = model.fit(Xtrain, args.iters, args.lr, args.batch_size)
    model.eval()
    model.save_something('final', Xtrain[::10])
    
    # Evaluate
    #elbo1, logpx1, parzen1 = model.evaluate(Xtrain)
    elbo2, logpx2, parzen2 = model.evaluate(Xtest)
    
    # Save results
    np.savez(direc + '/stats',
             elbo = loss[0], logpx = loss[1], kl = loss[2],
             #elbo1 = elbo1, logpx1 = logpx1, parzen1 = parzen1,
             elbo2 = elbo2, logpx2 = logpx2, parzen2 = parzen2
            )