import numpy as np
import tensorflow as tf


class GaussianAttention(tf.keras.layers.Layer):
    window_w: tf.Variable
    window_b: tf.Variable
    kappa: np.ndarray
    next_kappa: np.ndarray

    def __init__(self, k_mixtures, window_w_initializer, window_b_initializer, **kwargs):
        super(GaussianAttention, self).__init__(**kwargs)
        self.k_mixtures = k_mixtures
        self.window_w_initializer = window_w_initializer
        self.window_b_initializer = window_b_initializer

    def build(self, input_shape):
        batch = input_shape[0]
        hidden = input_shape[2]
        n_out = 3 * self.k_mixtures
        self.kappa = np.zeros((batch, self.k_mixtures, 1))
        self.window_w = self.add_weight("window_w", shape=[hidden, n_out], initializer=self.window_w_initializer)
        self.window_b = self.add_weight("window_b", shape=[n_out], initializer=self.window_b_initializer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'k_mixtures': self.k_mixtures,
            'window_w_initializer': self.window_w_initializer,
            'window_b_initializer': self.window_b_initializer,
        })
        return config

    def call(self, input0, **kwargs):
        assoc = tf.unstack(kwargs['stroke'], axis=1)
        result = tf.unstack(input0, axis=1)
        prev_kappa = self.kappa.copy()
        char_seq = kwargs['char']
        for i in range(len(result)):
            [alpha, beta, new_kappa] = self.get_window_params(result[i], prev_kappa)
            window, phi = self.get_window(alpha, beta, new_kappa, char_seq)
            result[i] = tf.concat((result[i], window, assoc[i]), 1)
            prev_kappa = new_kappa
        self.next_kappa = prev_kappa
        return tf.stack(result, axis=1)

    # ----- build the gaussian character window
    def get_window(self, alpha, beta, kappa, c):
        # phi -> [? x 1 x ascii_steps] and is a tf matrix
        # c -> [? x ascii_steps x alphabet] and is a tf matrix
        ascii_steps = c.get_shape()[1]  # number of items in sequence
        phi = self.get_phi(ascii_steps, alpha, beta, kappa)
        window = tf.matmul(phi, c)
        window = tf.squeeze(window, [1])  # window ~ [?,alphabet]
        return window, phi

    # get phi for all t,u (returns a [1 x tsteps] matrix) that defines the window
    @staticmethod
    def get_phi(ascii_steps, alpha, beta, kappa):
        # alpha, beta, kappa -> [?,kmixtures,1] and each is a tf variable
        u = np.linspace(0, ascii_steps - 1, ascii_steps)  # weight all the U items in the sequence
        kappa_term = tf.square(tf.subtract(kappa, u))
        exp_term = tf.multiply(-beta, kappa_term)
        phi_k = tf.multiply(alpha, tf.exp(exp_term))
        return tf.reduce_sum(input_tensor=phi_k, axis=1, keepdims=True)  # phi ~ [?,1,ascii_steps]

    def get_window_params(self, out_cell0, prev_kappa):
        abk_hats = tf.add(tf.matmul(out_cell0, self.window_w), self.window_b)  # abk_hats ~ [?,n_out]
        # abk_hats ~ [?,n_out] = "alpha, beta, kappa hats"
        abk = tf.exp(tf.reshape(abk_hats, [-1, 3 * self.k_mixtures, 1]))
        alpha, beta, kappa = tf.split(abk, 3, 1)  # alpha_hat, etc ~ [?,kmixtures]
        kappa = kappa + prev_kappa
        return alpha, beta, kappa  # each ~ [?,kmixtures,1]
