import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train, get_dataset_eval
from models import get_G, get_img_D, get_E, get_z_D
import random
import argparse
import math
import scipy.stats as stats
import tensorflow_probability as tfp


def KStest(real_z, fake_z):
    p_list = []
    for i in range(flags.batch_size_train):
        _, tmp_p = stats.ks_2samp(fake_z[i], real_z[i])
        p_list.append(tmp_p)
    return np.min(p_list), np.mean(p_list)


def train(con=False):
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset

    G = get_G([None, flags.z_dim])
    D = get_img_D([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    D_z = get_z_D([None, flags.z_dim])

    if con:
        G.load_weights('./checkpoint/{}/G.npz'.format(flags.param_dir))
        D.load_weights('./checkpoint/{}/D.npz'.format(flags.param_dir))
        E.load_weights('./checkpoint/{}/E.npz'.format(flags.param_dir))
        D_z.load_weights('./checkpoint/{}/Dz.npz'.format(flags.param_dir))

    G.train()
    D.train()
    E.train()
    D_z.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_G = flags.lr_G
    lr_E = flags.lr_E
    lr_D = flags.lr_D
    lr_Dz = flags.lr_Dz

    d_optimizer = tf.optimizers.Adam(lr_D, beta_1=flags.beta1, beta_2=flags.beta2)
    g_optimizer = tf.optimizers.Adam(lr_G, beta_1=flags.beta1, beta_2=flags.beta2)
    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    dz_optimizer = tf.optimizers.Adam(lr_Dz, beta_1=flags.beta1, beta_2=flags.beta2)

    tfd = tfp.distributions
    dist_normal = tfd.Normal(loc=0., scale=1.)
    dist_Bernoulli = tfd.Bernoulli(probs=0.5)
    dist_beta = tfd.Beta(0.5, 0.5)

    for step, batch_imgs in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        epoch_num = step // n_step_epoch
        # limit the epoch/step num
        if epoch_num > flags.n_epoch:
            break
        if step > flags.step_num:
            break
        # radius decay
        if epoch_num <= 20:
            flags.sigma = flags.sigma
        elif epoch_num <= 40:
            flags.sigma = flags.sigma / 2.0
        elif epoch_num <= 60:
            flags.sigma = flags.sigma / 2.0
        elif epoch_num <= 80:
            flags.sigma = flags.sigma / 2.0
        else:
            flags.sigma = flags.sigma / 2.0

        with tf.GradientTape(persistent=True) as tape:
            z = flags.scale * np.random.normal(loc=0.0, scale=flags.sigma * math.sqrt(flags.z_dim),
                                              size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            z += flags.scale * np.random.binomial(n=1, p=0.5,
                                                 size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)

            fake_z = E(batch_imgs)
            fake_imgs = G(fake_z)
            fake_logits = D(fake_imgs)
            real_logits = D(batch_imgs)
            fake_logits_z = D(G(z))
            real_z_logits = D_z(z)
            fake_z_logits = D_z(fake_z)
            
            e_loss_z = - tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
                       tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.ones_like(fake_z_logits))

            recon_loss = flags.lamba_recon * tl.cost.absolute_difference_error(batch_imgs, fake_imgs, is_mean=True)
            g_loss_x = - tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
                       tl.cost.sigmoid_cross_entropy(fake_logits, tf.ones_like(fake_logits))
            g_loss_z = - tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z)) + \
                       tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.ones_like(fake_logits_z))
            e_loss = recon_loss + e_loss_z
            g_loss = recon_loss + g_loss_x + g_loss_z

            d_loss = tl.cost.sigmoid_cross_entropy(real_logits, tf.ones_like(real_logits)) + \
                     tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
                     tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z))

            dz_loss = tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
                      tl.cost.sigmoid_cross_entropy(real_z_logits, tf.ones_like(real_z_logits))


        # Updating Encoder
        grad = tape.gradient(e_loss, E.trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E.trainable_weights))

        # Updating Generator
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

        # Updating Discriminator
        grad = tape.gradient(d_loss, D.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

        # Updating D_z & D_h
        grad = tape.gradient(dz_loss, D_z.trainable_weights)
        dz_optimizer.apply_gradients(zip(grad, D_z.trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] e_loss: {:.5f}, g_loss: {:.5f}, d_loss: {:.5f}, "
                  "dz_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, e_loss, g_loss,
                   d_loss, dz_loss))
            # Kstest
            p_min, p_avg = KStest(z, fake_z)
            print("kstest: min:{}, avg:{}", p_min, p_avg)

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G.save_weights('{}/{}/G.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D.save_weights('{}/{}/D.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E.save_weights('{}/{}/E.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_z.save_weights('{}/{}/Dz.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            # z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            z = flags.scale * np.random.normal(loc=0.0, scale=flags.sigma * math.sqrt(flags.z_dim),
                                               size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            z += flags.scale * np.random.binomial(n=1, p=0.5,
                                                  size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            result = G(z)
            G.train()
            tl.visualize.save_images(result.numpy(), [8, 8],
                                     '{}/{}/train_{:02d}_{:04d}.png'.format(flags.sample_dir, flags.param_dir,
                                                                            step // n_step_epoch, step))
        del tape


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train(con=args.is_continue)
