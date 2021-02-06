import tensorflow as tf
from utils import *


# noinspection PyAttributeOutsideInit
class GaussianAttention(tf.keras.layers.Layer):
    def __init__(self, kmixtures, window_w_initializer, window_b_initializer):
        super(GaussianAttention, self).__init__()
        self.kmixtures = kmixtures
        self.window_w_initializer = window_w_initializer
        self.window_b_initializer = window_b_initializer

    def build(self, input_shape):
        batch = input_shape[0]
        hidden = input_shape[2]
        n_out = 3 * self.kmixtures
        self.init_kappa = self.add_weight("init_kappa", shape=[batch, self.kmixtures, 1])
        self.window_w = self.add_weight("window_w", shape=[hidden, n_out], initializer=self.window_w_initializer)
        self.window_b = self.add_weight("window_b", shape=[n_out], initializer=self.window_b_initializer)

    def call(self, input0, **kwargs):
        assoc = tf.unstack(kwargs['stroke'], axis=1)
        result = tf.unstack(input0, axis=1)
        prev_kappa = self.init_kappa
        char_seq = kwargs['char']
        for i in range(len(result)):
            [alpha, beta, new_kappa] = self.get_window_params(result[i], prev_kappa)
            window, phi = self.get_window(alpha, beta, new_kappa, char_seq)
            result[i] = tf.concat((result[i], window, assoc[i]), 1)
            prev_kappa = new_kappa
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
    def get_phi(self, ascii_steps, alpha, beta, kappa):
        # alpha, beta, kappa -> [?,kmixtures,1] and each is a tf variable
        u = np.linspace(0, ascii_steps - 1, ascii_steps)  # weight all the U items in the sequence
        kappa_term = tf.square(tf.subtract(kappa, u))
        exp_term = tf.multiply(-beta, kappa_term)
        phi_k = tf.multiply(alpha, tf.exp(exp_term))
        return tf.reduce_sum(input_tensor=phi_k, axis=1, keepdims=True)  # phi ~ [?,1,ascii_steps]

    def get_window_params(self, out_cell0, prev_kappa):
        abk_hats = tf.add(tf.matmul(out_cell0, self.window_w), self.window_b)  # abk_hats ~ [?,n_out]
        # abk_hats ~ [?,n_out] = "alpha, beta, kappa hats"
        abk = tf.exp(tf.reshape(abk_hats, [-1, 3 * self.kmixtures, 1]))
        alpha, beta, kappa = tf.split(abk, 3, 1)  # alpha_hat, etc ~ [?,kmixtures]
        kappa = kappa + prev_kappa
        return alpha, beta, kappa  # each ~ [?,kmixtures,1]


# noinspection PyAttributeOutsideInit
class MDN(tf.keras.layers.Layer):
    def __init__(self, rnn_size, nmixtures, initializer):
        super(MDN, self).__init__()
        self.rnn_size = rnn_size
        self.nmixtures = nmixtures
        self.initializer = initializer

    def build(self, input_shape):
        n_out = 1 + self.nmixtures * 6  # params = end_of_stroke + 6 parameters per Gaussian
        self.mdn_w = self.add_weight("output_w", shape=[self.rnn_size, n_out], initializer=self.initializer)
        self.mdn_b = self.add_weight("output_b", shape=[n_out], initializer=self.initializer)

    def call(self, input0, **kwargs):
        flattened = tf.reshape(tf.concat(input0, 0), [-1, self.rnn_size])  # concat outputs for efficiency
        dense = tf.add(tf.matmul(flattened, self.mdn_w), self.mdn_b)
        return tf.stack(dense)


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


class Model:
    def __init__(self, args, logger):
        self.logger = logger

        # ----- transfer some of the args params over to the model

        # model params
        self.train = args.train
        self.batch_size = args.batch_size if self.train else 1  # training/sampling specific
        self.tsteps = args.tsteps if self.train else 1  # training/sampling specific
        self.alphabet = args.alphabet
        # training params
        self.grad_clip = args.grad_clip
        # misc
        self.tsteps_per_ascii = args.tsteps_per_ascii
        self.data_dir = args.data_dir

        self.logger.write('\tusing alphabet{}'.format(self.alphabet))
        self.char_vec_len = len(self.alphabet) + 1  # plus one for <UNK> token
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)

        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)
        window_b_initializer = tf.keras.initializers.TruncatedNormal(mean=-3.0, stddev=0.25)

        # define the network layers
        cell0 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer, dropout=args.dropout
        )
        cell1 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer, dropout=args.dropout
        )
        cell2 = tf.keras.layers.LSTM(
            args.rnn_size, return_sequences=True, kernel_initializer=graves_initializer, dropout=args.dropout
        )
        attention = GaussianAttention(args.kmixtures, graves_initializer, window_b_initializer)
        mdn = MDN(args.rnn_size, args.nmixtures, graves_initializer)

        # link the network
        model_stroke = tf.keras.layers.Input(
            name='stroke', shape=(self.tsteps, 3,), batch_size=self.batch_size
        )
        model_char = tf.keras.layers.Input(
            name='char', shape=(self.ascii_steps, self.char_vec_len,), batch_size=self.batch_size
        )
        model_out = mdn(cell2(cell1(attention(cell0(model_stroke), stroke=model_stroke, char=model_char))))
        self.model = tf.keras.Model([model_stroke, model_char], model_out)
        self.model.summary()
        # self.cost = loss / (self.batch_size * self.tsteps)

    def setup(self, optimizer, learning_rate, decay, momentum, lr_decay, epoch_size):
        rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, epoch_size, lr_decay)
        if optimizer == 'adam':
            s_optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
        elif optimizer == 'rmsprop':
            s_optimizer = tf.keras.optimizers.RMSprop(learning_rate=rate, rho=decay, momentum=momentum)
        else:
            raise ValueError("Optimizer type not recognized")
        self.model.compile(optimizer=s_optimizer, loss=tf2_loss)

    def train_network(self, train, validation, epochs):
        # model training setup
        self.model.fit(
            x=train,
            validation_data=validation,
            epochs=epochs,
        )

    # ----- for restoring previous models
    def try_load_model(self, save_path):
        load_was_success = True  # yes, I'm being optimistic
        global_step = 0
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            self.logger.write("no saved model to load. starting new session")
            load_was_success = False
        else:
            self.logger.write("loaded model: {}".format(load_path))
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            global_step = int(load_path.split('-')[-1])
        return load_was_success, global_step
