import argparse

from model.train import Training
from model.utils import set_path


def main():
    parser = argparse.ArgumentParser()

    # general model params
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
    parser.add_argument('--log_dir', type=str, default='./logs', help='location, relative to execution, of log files')
    parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='location, relative to execution, of cache')
    parser.add_argument('--save_path', type=str, default='./saved', help='location to save model')
    parser.add_argument('--board_path', type=str, default="./tb_logs/", help='location, relative to execution, board')

    args = parser.parse_args()
    set_path(base='', data=args.data_dir, cache=args.cache_dir, model_obj=args.save_path)
    Training(args).train()


if __name__ == '__main__':
    main()
