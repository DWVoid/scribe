from model.model import Model
from utils.logger import LoggerRoot


class Sample:
    def __init__(self, args) -> None:
        self.args = args
        self.logger = LoggerRoot(args.log_dir)
        self.logger.write('{}'.format(args))
        self.model = Model()


    def __eval_one(self):
        fn = self.model.get()

