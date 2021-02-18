import argparse

from model.stat import Stats
from model.utils import set_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs', help='location, relative to execution, of log files')
    parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='location, relative to execution, of cache')
    parser.add_argument('--save_path', type=str, default='./saved', help='location to save model')
    args = parser.parse_args()
    set_path(base='', data=args.data_dir, cache=args.cache_dir, model_obj=args.save_path)
    Stats(args).compute().visualize()


if __name__ == '__main__':
    main()
