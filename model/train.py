import concurrent.futures as cf
import threading
import os
from typing import *

import tensorflow as tf

import model.utils as du
from model.types import *
from model.source import DataSource
from model.model import Model
from utils.logger import Logger, LoggerRoot


class Training:
    logger: Logger

    def __init__(self, args) -> None:
        self.logger = LoggerRoot(args.log_dir)
        self.logger.write('{}'.format(args))
        self.logger.write("loading data...")
        self.data = DataSource(args, logger=Logger(self.logger))
        self.all_weights_old = du.load_model_obj('all_weights', dict())
        self.threadLocal = threading.local()
        # transfer some model settings
        self.batch_size = args.batch_size
        self.n_epochs = args.nepochs
        self.board_path = args.board_path
        # record the args
        self.args = args

    def train(self) -> None:
        all_weights = dict()
        self.logger.write('training')
        with cf.ThreadPoolExecutor() as executor:
            for writer, weighs in executor.map(self.apply_train_one, self.data.datasets()):
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

    def get_thread_model_obj(self):
        model = getattr(self.threadLocal, 'model', None)
        if model is None:
            setattr(self.threadLocal, 'model', self.gen_model_object())
        return getattr(self.threadLocal, 'model', None)

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
            model, weights = self.get_thread_model_obj()
            model.set_weights(weights)
            train, validation = dataset
            logger.write("training model...")
            model.train_network(
                train=tf.data.Dataset.from_tensor_slices(train).batch(self.batch_size, drop_remainder=True),
                validation=tf.data.Dataset.from_tensor_slices(validation).batch(self.batch_size, drop_remainder=True),
                epochs=self.n_epochs,
                tensorboard_logs=self.board_path
            )
            logger.write("saving model...")
            model.save_model(du.model_obj_path(str(writer)))
            weights = model.get_weights()
            du.save_model_obj(weights_path, weights)
            return weights
