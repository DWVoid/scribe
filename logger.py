import os


class Logger:
    def __init__(self, args):
        self.path = os.path.join(args.log_dir, 'train_scribe.txt' if args.train else 'sample_scribe.txt')
        with open(self.path, 'w') as f:
            f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print(s)
        with open(self.path, 'a') as f:
            f.write(s + '\n')