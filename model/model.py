import tensorflow as tf
import numpy as np
from model.source import *

from model.gaussian_attention import GaussianAttention
from model.mdn import MDN


# transform dense NN outputs into params for MDN
def get_mdn_coef(dense):
    # returns the tf slices containing mdn dist params (eq 18...23 of http://arxiv.org/abs/1308.0850)
    eos_hat = dense[:, 0:1]  # end of sentence tokens
    pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(dense[:, 1:], 6, 1)
    eos = tf.sigmoid(-1 * eos_hat)  # technically we gained a negative sign
    pi = tf.nn.softmax(pi_hat)  # softmax z_pi:
    sigma1 = tf.exp(sigma1_hat)
    sigma2 = tf.exp(sigma2_hat)  # exp for sigmas
    rho = tf.tanh(rho_hat)  # tanh for rho (squish between -1 and 1)
    return [pi, pi_hat, sigma1, sigma2, sigma1_hat, sigma2_hat, eos, mu1, mu2, rho]


# define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
    x_mu1 = tf.subtract(x1, mu1)
    x_mu2 = tf.subtract(x2, mu2)
    Z = tf.square(tf.divide(x_mu1, s1)) + \
        tf.square(tf.divide(x_mu2, s2)) - \
        2 * tf.divide(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
    rho_square_term = 1 - tf.square(rho)
    power_e = tf.exp(tf.divide(-Z, 2 * rho_square_term))
    regularize_term = 2 * np.pi * tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
    gaussian = tf.divide(power_e, regularize_term)
    return gaussian


# define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
def get_loss(pi, x1_data, x2_data, eos_data, mu1, mu2, sigma1, sigma2, rho, eos):
    gaussian = gaussian2d(x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho)
    term1 = tf.multiply(gaussian, pi)
    term1 = tf.reduce_sum(input_tensor=term1, axis=1, keepdims=True)  # do inner summation
    term1 = -tf.math.log(tf.maximum(term1, 1e-20))  # some errors are zero -> numerical errors.
    term2 = tf.multiply(eos, eos_data) + tf.multiply(1 - eos, 1 - eos_data)  # modified Bernoulli -> eos probability
    term2 = -tf.math.log(term2)  # negative log error gives loss
    return tf.reduce_sum(input_tensor=term1 + term2)  # do outer summation


def tf2_loss(y_true, y_pred):
    [pi, _, sigma1, sigma2, _, _, eos, mu1, mu2, rho] = get_mdn_coef(y_pred)
    [x1_data, x2_data, eos_data] = tf.split(tf.reshape(y_true, [-1, 3]), 3, 1)
    return get_loss(pi, x1_data, x2_data, eos_data, mu1, mu2, sigma1, sigma2, rho, eos)


# noinspection PyAttributeOutsideInit
class Model:
    attention: GaussianAttention = None

    def __init__(self, logger):
        self.logger = logger

    def build(self, args, train=True):
        # model params
        self.batch_size = args.batch_size if train else 1
        self.tsteps = args.tsteps if train else 1
        # misc
        self.tsteps_per_ascii = args.tsteps_per_ascii

        self.logger.write('\tusing alphabet{}'.format(args.alphabet))
        char_vec_len = len(args.alphabet) + 1  # plus one for <UNK> token
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)

        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)
        window_b_initializer = tf.keras.initializers.TruncatedNormal(mean=-3.0, stddev=0.25)

        # define the network layers
        cell0 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer,
            dropout=args.dropout if train else 0.
        )
        cell1 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer,
            dropout=args.dropout if train else 0.
        )
        cell2 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer,
            dropout=args.dropout if train else 0.
        )
        self.attention = GaussianAttention(args.kmixtures, graves_initializer, window_b_initializer)
        mdn = MDN(args.rnn_size, args.nmixtures, graves_initializer)

        # link the network
        model_stroke = tf.keras.layers.Input(
            name='stroke', shape=(self.tsteps, 3,), batch_size=self.batch_size
        )
        model_char = tf.keras.layers.Input(
            name='char', shape=(self.ascii_steps, char_vec_len,), batch_size=self.batch_size
        )
        model_out = mdn(cell2(cell1(self.attention(cell0(model_stroke), stroke=model_stroke, char=model_char))))
        self.model = tf.keras.Model([model_stroke, model_char], model_out)
        # self.cost = loss / (self.batch_size * self.tsteps)

        if train:
            # define the training parameters and prepare for training
            # training params
            self.grad_clip = args.grad_clip
            size = args.nbatches * args.batch_size
            rate = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, size, args.lr_decay)
            if args.optimizer == 'adam':
                s_optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
            elif args.optimizer == 'rmsprop':
                s_optimizer = tf.keras.optimizers.RMSprop(learning_rate=rate, rho=args.decay, momentum=args.momentum)
            else:
                raise ValueError("Optimizer type not recognized")
            self.model.compile(optimizer=s_optimizer, loss=tf2_loss)

    def duplicate(self):
        model = Model(Logger(self.logger))
        model.model = tf.keras.models.clone_model(self.model)
        return model

    def train_network(self, train, validation, epochs, tensorboard_logs):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)
        # model training setup
        self.model.fit(
            x=train,
            validation_data=validation,
            epochs=epochs,
            callbacks=[tensorboard_callback],
            verbose=2
        )

    def save_weights(self, save_path):
        return self.model.save_weights(save_path)

    def load_weights(self, save_path):
        self.model.load_weights(save_path)

    def save_model(self, save_path):
        self.model.save(filepath=save_path)

    # ----- for restoring previous models
    def try_load_model(self, save_path, compile=True):
        try:
            self.model = tf.keras.models.load_model(filepath=save_path, compile=compile)
        except IOError:
            self.logger.write("no saved model to load. starting new session")
            return True
        else:
            return False

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get(self):
        return self.model

    def get_attention(self) -> GaussianAttention:
        return self.attention