import tensorflow as tf
import argparse
import time

from model3 import Model
from sample import DataSource, Logger


def main():
    parser = argparse.ArgumentParser()

    # general model params
    parser.add_argument('--train', dest='train', action='store_true', help='train the model')
    parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
    parser.add_argument('--rnn_size', type=int, default=100, help='size of RNN hidden state')
    parser.add_argument('--tsteps', type=int, default=150, help='RNN time steps (for backprop)')
    parser.add_argument('--nmixtures', type=int, default=8, help='number of gaussian mixtures')

    # window params
    parser.add_argument('--kmixtures', type=int, default=1, help='number of gaussian mixtures for character window')
    parser.add_argument('--alphabet', type=str, default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        help='default is a-z, A-Z, space, and <UNK> tag')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25, help='expected number of pen points per character')

    # training params
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for each gradient step')
    parser.add_argument('--nbatches', type=int, default=10, help='number of batches per epoch')
    parser.add_argument('--nepochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.85, help='probability of keeping neuron during dropout')

    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help="ctype of optimizer: 'rmsprop' 'adam'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate for learning rate')
    parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for rmsprop')

    # book-keeping
    parser.add_argument('--data_scale', type=int, default=50, help='amount to scale data down before training')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='location, relative to execution, of log files')
    parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='location, relative to execution, of cache')
    parser.add_argument('--save_path', type=str, default='saved/model.ckpt', help='location to save model')
    parser.add_argument('--save_every', type=int, default=500, help='number of batches between each save')
    parser.add_argument('--board_path', type=str, default="./tb_logs/", help='location, relative to execution, board')

    # sampling
    parser.add_argument('--text', type=str, default='', help='string for sampling model (defaults to test cases)')
    parser.add_argument('--style', type=int, default=-1,
                        help='optionally condition model on a preset style (using data in styles.p)')
    parser.add_argument('--bias', type=float, default=1.0,
                        help='higher bias means neater, lower means more diverse (range is 0-5)')
    parser.add_argument('--sleep_time', type=int, default=60 * 5, help='time to sleep between running sampler')
    parser.set_defaults(train=True)
    args = parser.parse_args()

    train_model(args) if args.train else sample_model(args)


def train_model(args):
    logger = Logger(args)  # make logging utility
    logger.write("\nTRAINING MODE...")
    logger.write("{}\n".format(args))
    logger.write("loading data...")
    data_loader = DataSource(args, logger=logger)
    model = Model(logger=logger)
    logger.write("building model...")
    model.build(args)
    model.save_weights(args.save_path + 'raw')
    logger.write("training...")
    for writer, dataset in data_loader.datasets():
        logger.write("reloading model...")
        model.load_weights(args.save_path + 'raw')
        train, validation = dataset
        model.train_network(
            train=tf.data.Dataset.from_tensor_slices(train).batch(args.batch_size, drop_remainder=True),
            validation=tf.data.Dataset.from_tensor_slices(validation).batch(args.batch_size, drop_remainder=True),
            epochs=args.nepochs,
            tensorboard_logs=args.board_path
        )
        logger.write("saving model...")
        model.save_model(args.save_path+str(writer))


def sample_model(args, logger=None):
    if args.text == '':
        strings = ['call me ishmael some years ago', 'A project by Sam Greydanus', 'mmm mmm mmm mmm mmm mmm mmm',
                   'What I cannot create I do not understand', 'You know nothing Jon Snow']  # test strings
    else:
        strings = [args.text]

    logger = Logger(args) if logger is None else logger  # instantiate logger, if None
    logger.write("\nSAMPLING MODE...")
    logger.write("loading data...")

    logger.write("building model...")
    model = Model(logger)

    logger.write("attempt to load saved model...")
    load_was_success, global_step = model.try_load_model(args.save_path)

    if load_was_success:
        for s in strings:
            strokes, phis, windows, kappas = sample(s, model, args)

            w_save_path = '{}figures/iter-{}-w-{}'.format(args.log_dir, global_step, s[:10].replace(' ', '_'))
            g_save_path = '{}figures/iter-{}-g-{}'.format(args.log_dir, global_step, s[:10].replace(' ', '_'))
            l_save_path = '{}figures/iter-{}-l-{}'.format(args.log_dir, global_step, s[:10].replace(' ', '_'))

            window_plots(phis, windows, save_path=w_save_path)
            gauss_plot(strokes, 'Heatmap for "{}"'.format(s), figsize=(2 * len(s), 4), save_path=g_save_path)
            line_plot(strokes, 'Line plot for "{}"'.format(s), figsize=(len(s), 2), save_path=l_save_path)

            # make sure that kappas are reasonable
            logger.write("kappas: \n{}".format(str(kappas[min(kappas.shape[0] - 1, args.tsteps_per_ascii), :])))
    else:
        logger.write("load failed, sampling canceled")

    if True:
        tf.compat.v1.reset_default_graph()
        time.sleep(args.sleep_time)
        sample_model(args, logger=logger)


if __name__ == '__main__':
    main()
