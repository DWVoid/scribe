import os
import tensorflow as tf
import model.utils as du
from model.types import *
from model.source import DataSource
from model.model import Model
from utils.logger import Logger, LoggerRoot


class Training:
    logger: Logger

    def __init__(self, args) -> None:
        self.args = args
        self.logger = LoggerRoot(args.log_dir)
        self.logger.write('{}'.format(args))
        self.logger.write("loading data...")
        self.data = DataSource(args, logger=Logger(self.logger))
        self.all_weights_old = du.load_model_obj('all_weights', dict())
        self.model, self.weights = self.gen_model_object()
        # transfer some model settings
        self.batch_size = args.batch_size
        self.n_epochs = args.nepochs
        self.board_path = args.board_path

    def train(self) -> None:
        all_weights = dict()
        self.logger.write('training')
        for writer, weighs in map(self.apply_train_one, self.data.datasets()):
            self.logger.write('collecting writer {}'.format(writer))
            all_weights[writer] = weighs
        self.all_weights_old = all_weights
        du.save_model_obj('all_weights', self.all_weights_old)

    def gen_model_object(self):
        self.logger.write("building model...")
        model = Model(logger=Logger(self.logger))
        model.build(self.args)
        return model, model.get_weights()

    def apply_train_one(self, o) -> Any:
        writer, dataset = o
        return writer, self.train_one(writer, dataset)

    def train_one(self, writer: int, dataset: DataSetCompiled) -> Any:
        logger = Logger(self.logger)
        # skip if the weight is in the map
        if writer in self.all_weights_old:
            logger.write('data of writer {} has been trained'.format(writer))
            return self.all_weights_old[writer]
        weights_path = os.path.join('weights', str(writer))
        weights = du.load_model_obj(weights_path)
        if weights is not None:
            logger.write('data of writer {} has been trained, transferring to map'.format(writer))
            return weights
        else:
            logger.write("training start on writer: {}".format(writer))
            logger.write("resetting weights...")
            self.model.set_weights(weights)
            train, validation = dataset
            logger.write("training model...")
            self.model.train_network(
                train=tf.data.Dataset.from_tensor_slices(train).batch(self.batch_size, drop_remainder=True),
                validation=tf.data.Dataset.from_tensor_slices(validation).batch(self.batch_size, drop_remainder=True),
                epochs=self.n_epochs,
                tensorboard_logs=self.board_path
            )
            logger.write("saving model...")
            weights = self.model.get_weights()
            du.save_model_obj(weights_path, weights)
            return weights
