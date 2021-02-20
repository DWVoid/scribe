from data import *
import tensorflow as tf
import model.utils as du
import numpy as np
import argparse
import pickle
import time
import os

from model.model import get_mdn_coef, Model
from utils.logger import Logger, LoggerRoot
from model.utils import set_path


def main():
    parser = argparse.ArgumentParser()

    # general model params
    parser.add_argument('--rnn_size', type=int, default=100, help='size of RNN hidden state')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for each gradient step')
    parser.add_argument('--tsteps', type=int, default=1000, help='RNN time steps (for backprop)')
    parser.add_argument('--nmixtures', type=int, default=8, help='number of gaussian mixtures')

    # window params
    parser.add_argument('--kmixtures', type=int, default=1, help='number of gaussian mixtures for character window')
    parser.add_argument('--alphabet', type=str, default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        help='default is a-z, A-Z, space, and <UNK> tag')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25, help='expected number of pen points per character')

    # book-keeping
    parser.add_argument('--log_dir', type=str, default='./logs', help='location, relative to execution, of log files')
    parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='location, relative to execution, of cache')
    parser.add_argument('--save_path', type=str, default='./saved', help='location to save model')
    parser.add_argument('--board_path', type=str, default="./tb_logs/", help='location, relative to execution, board')

    # sampling
    parser.add_argument('--text', type=str, default='', help='string for sampling model (defaults to test cases)')
    parser.add_argument('--style', type=int, default=-1,
                        help='optionally condition model on a preset style (using data in styles.p)')
    parser.add_argument('--bias', type=float, default=1.0,
                        help='higher bias means neater, lower means more diverse (range is 0-5)')
    parser.add_argument('--sleep_time', type=int, default=60 * 5, help='time to sleep between running sampler')
    parser.add_argument('--writer', type=int, default=10000, help='writer id to load')
    parser.set_defaults(train=True)
    args = parser.parse_args()
    set_path(base='', data=args.data_dir, cache=args.cache_dir, model_obj=args.save_path)
    sample_model(args)


def sample_model(args):
    if args.text == '':
        strings = ['call me ishmael some years ago', 'A project by Sam Greydanus', 'mmm mmm mmm mmm mmm mmm mmm',
                   'What I cannot create I do not understand', 'You know nothing Jon Snow']  # test strings
    else:
        strings = [args.text]

    logger = LoggerRoot(args.log_dir)
    logger.write("\nSAMPLING MODE...")
    logger.write("loading data...")

    logger.write("building model...")
    model = Model(logger)
    model.build(args, train=False)

    logger.write("attempt to load saved model...")
    model.set_weights(du.load_model_obj(os.path.join('weights', str(args.writer))))

    for s in strings:
        strokes = sample(s, model, args)

        g_save_path = '{}figures/iter-{}-g-{}'.format(args.log_dir, 1, s[:10].replace(' ', '_'))
        l_save_path = '{}figures/iter-{}-l-{}'.format(args.log_dir, 1, s[:10].replace(' ', '_'))

        gauss_plot(strokes, 'Heatmap for "{}"'.format(s), figsize=(2 * len(s), 4), save_path=g_save_path)
        line_plot(strokes, 'Line plot for "{}"'.format(s), figsize=(len(s), 2), save_path=l_save_path)

    if True:
        tf.compat.v1.reset_default_graph()
        time.sleep(args.sleep_time)
        sample_model(args)


def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def get_style_states(model, args):
    if args.style == -1:
        return  # model 'chooses' random style

    with open(os.path.join(args.data_dir, 'styles.p'), 'r') as f:
        style_strokes, style_strings = pickle.load(f)

    style_strokes, style_string = style_strokes[args.style], style_strings[args.style]
    style_onehot = [to_one_hot(style_string, model.ascii_steps, args.alphabet)]

    style_stroke = np.zeros((1, 1, 3), dtype=np.float32)
    style_kappa = np.zeros((1, args.k_mixtures, 1))
    prime_len = 500  # must be <= 700

    for i in range(prime_len):
        style_stroke[0][0] = style_strokes[i, :]
        model.get_attention().kappa = style_kappa
        model.get()(style_stroke, style_onehot)
        style_kappa = model.get_attention().next_kappa
    # TODO: only the c vectors should be primed


def sample(input_text, model, args):
    # initialize some parameters
    one_hot = [to_one_hot(input_text, model.ascii_steps, args.alphabet)]  # convert input string to one-hot vector
    get_style_states(model, args)  # get numpy zeros states for all three LSTMs
    kappa = np.zeros((1, args.kmixtures, 1))  # attention mechanism's read head should start at index 0
    prev_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)  # start with a pen stroke at (0,0)

    # the data we're going to generate will go here
    strokes = []

    finished = False
    i = 0
    while not finished:
        model.get_attention().kappa = kappa
        dense = model.get()(prev_x, one_hot)
        kappa = model.get_attention().next_kappa
        [_, pi_hat, _, _, sigma1_hat, sigma2_hat, eos, mu1, mu2, rho] = get_mdn_coef(dense)

        # bias stuff:
        sigma1 = np.exp(sigma1_hat - args.bias)
        sigma2 = np.exp(sigma2_hat - args.bias)
        pi_hat *= 1 + args.bias  # apply bias
        pi = np.zeros_like(pi_hat)  # need to preallocate
        pi[0] = np.exp(pi_hat[0]) / np.sum(np.exp(pi_hat[0]), axis=0)  # softmax

        # choose a component from the MDN
        idx = np.random.choice(pi.shape[1], p=pi[0])
        eos = 1 if 0.35 < eos[0][0] else 0  # use 0.5 as arbitrary boundary
        x1, x2 = sample_gaussian2d(mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx])

        # store the info at this time step
        strokes.append([mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx], eos])

        # test if finished (has the read head seen the whole ascii sequence?)
        # main_kappa_idx = np.where(alpha[0]==np.max(alpha[0]));
        # finished = True if kappa[0][main_kappa_idx] > len(input_text) else False
        finished = True if i > args.t_steps else False

        # new input is previous output
        prev_x[0][0] = np.array([x1, x2, eos], dtype=np.float32)
        i += 1

    strokes = np.vstack(strokes)

    # the network predicts the displacements between pen points, so do a running sum over the time dimension
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
    return strokes


# a heatmap for the probabilities of each pen point in the sequence
def gauss_plot(strokes, title, figsize=(20, 2), save_path='.'):
    import matplotlib.mlab as mlab
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)  #
    buff = 1
    epsilon = 1e-4
    minx, maxx = np.min(strokes[:, 0]) - buff, np.max(strokes[:, 0]) + buff
    miny, maxy = np.min(strokes[:, 1]) - buff, np.max(strokes[:, 1]) + buff
    delta = abs(maxx - minx) / 400.

    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(strokes.shape[0]):
        gauss = mlab.bivariate_normal(X, Y, mux=strokes[i, 0], muy=strokes[i, 1],
                                      sigmax=strokes[i, 2], sigmay=strokes[i, 3],
                                      sigmaxy=0)  # sigmaxy=strokes[i,4] gives error
        Z += gauss / (np.max(gauss) + epsilon)

    plt.title(title, fontsize=20)
    plt.imshow(Z)
    plt.savefig(save_path)
    plt.clf()
    plt.cla()


# plots the stroke data (handwriting!)
def line_plot(strokes, title, figsize=(20, 2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:, -1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1]  # add start and end indices
    for i in range(len(eos_preds) - 1):
        start = eos_preds[i] + 1
        stop = eos_preds[i + 1]
        plt.plot(strokes[start:stop, 0], strokes[start:stop, 1], 'b-', linewidth=2.0)  # draw a stroke
    plt.title(title, fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf()
    plt.cla()


# index position 0 means "unknown"
def to_one_hot(s: str, ascii_steps: int, alphabet: str) -> np.ndarray:
    s = s[:3e3] if len(s) > 3e3 else s  # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        seq = seq[:ascii_steps]
    else:
        seq = seq + [0] * (ascii_steps - len(seq))
    one_hot = np.zeros((ascii_steps, len(alphabet) + 1), dtype=np.float32)
    one_hot[np.arange(ascii_steps), seq] = 1
    return one_hot


if __name__ == '__main__':
    main()
