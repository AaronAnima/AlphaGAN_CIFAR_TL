import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from config import flags
from utils import WeightNorm


def get_G(shape_z, gf_dim=64):    # Dimension of gen filters in first conv layer. [64]
    # # input: (100,)
    # w_init = tf.random_normal_initializer(stddev=0.02)
    # gamma_init = tf.random_normal_initializer(1., 0.02)
    # nz = Input(shape_z)
    # n = Dense(n_units=3136, act=tf.nn.relu, W_init=w_init)(nz)
    # n = Reshape(shape=[-1, 14, 14, 16])(n)
    # n = DeConv2d(64, (5, 5), strides=(2, 2), W_init=w_init, b_init=None)(n) # (1, 28, 28, 64)
    # n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)
    # n = DeConv2d(flags.c_dim, (5, 5), strides=(1, 1), padding="VALID", W_init=w_init, b_init=None)(n) # (1, 32, 32, 3)
    # return tl.models.Model(inputs=nz, outputs=n, name='generator')

    image_size = 32
    s16 = image_size // 16
    # w_init = tf.glorot_normal_initializer()
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    ni = Input(shape_z)
    nn = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, s16, s16, gf_dim * 8])(nn) # [-1, 2, 2, gf_dim * 8]
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 4, 4, gf_dim * 4]
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 8, 8, gf_dim * 2]
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 16, 16, gf_dim *]
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(3, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn) # [-1, 32, 32, 3]

    return tl.models.Model(inputs=ni, outputs=nn, name='generator')


def get_img_D(shape):
    df_dim = 8
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ni = Input(shape)
    n = Conv2d(df_dim, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    nf = Flatten(name='flatten')(n)
    n = Dense(n_units=1, act=None, W_init=w_init)(nf)
    return tl.models.Model(inputs=ni, outputs=n, name='img_Discriminator')


def get_E(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ni = Input(shape)   # (1, 64, 64, 3)
    n = Conv2d(3, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)  # (1, 16, 16, 3)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(32, (5, 5), (1, 1), padding="VALID", act=None, W_init=w_init, b_init=None)(n)  # (1, 12, 12, 32)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(64, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)  # (1, 6, 6, 64)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Flatten(name='flatten')(n)
    nz = Dense(n_units=flags.z_dim, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=ni, outputs=nz, name='encoder')


def get_z_D(shape_z):
    gamma_init = tf.random_normal_initializer(1., 0.02)
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    nz = Input(shape_z)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(nz)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=1, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=nz, outputs=n, name='c_Discriminator')
