import os
import tensorflow as tf
import model.utils as du
from model.types import *
from model.model import Model
from model.source import DataSource
from utils.logger import Logger, LoggerRoot


class Training:
    logger: Logger

    def __init__(self, args) -> None:
        self.args = args
        self.logger = LoggerRoot(args.log_dir)
        self.logger.write('{}'.format(args))
        # check if the result exists. if is, print a hint and exit immediately
        if du.load_model_obj('all_weights', None) is not None:
            self.logger.write("training has completed. delete everything under saved directory if you want re-train")
            self.data = None
        else:
            # transfer some model settings
            self.batch_size = args.batch_size
            self.n_epochs = args.nepochs
            self.board_path = args.board_path
            # prepare data and training
            self.logger.write("loading data...")
            self.data = DataSource(args, logger=Logger(self.logger)).datasets()
            self.logger.write("counting progress...")
            self.all_weights = self.__init_count_progress()
            self.__init_trim_datasets()
            self.logger.write("compiling model...")
            self.model, self.zero_weights = self.__init_model_object()

    def __init_count_progress(self) -> Dict[int, List[np.ndarray]]:
        all_weights: Dict[int, List[np.ndarray]] = dict()
        if not os.path.exists(du.model_obj_path('weights')):
            os.mkdir(du.model_obj_path('weights'))
        else:
            for writer in self.data.keys():
                weights = du.load_model_obj(os.path.join('weights', str(writer)), None)
                if weights is not None:
                    all_weights[writer] = weights
        return all_weights

    def __init_trim_datasets(self) -> None:
        if bool(self.all_weights):
            self.logger.write('some writers are already trained, involving')
            self.logger.write('{}'.format(self.all_weights.keys()))
            self.logger.write('trimming training datasets...')
            for writer in self.all_weights.keys():
                self.data.pop(writer)

    def __init_model_object(self):
        model = Model(logger=Logger(self.logger))
        model.build(self.args)
        return model, model.get_weights()

    def train(self) -> None:
        if self.data is not None:
            self.logger.write('training...')
            for writer, weights in map(self.__apply_train_one, self.data.items()):
                self.all_weights[writer] = weights
            du.save_model_obj('all_weights', self.all_weights)
            self.logger.write('training completed')

    def __apply_train_one(self, o: Tuple[int, DataSetCompiled]) -> Tuple[int, List[np.ndarray]]:
        (writer, (train, validation)) = o
        return writer, self.__train_one(writer, train, validation)

    def __train_one(self, writer: int, train: DataSetShaped, validation: DataSetShaped) -> List[np.ndarray]:
        logger = Logger(self.logger)
        logger.write("on writer: {}, resetting weights...".format(writer))
        self.model.set_weights(self.zero_weights)
        logger.write("training model...")
        self.model.train_network(
            train=tf.data.Dataset.from_tensor_slices(train).batch(self.batch_size, drop_remainder=True),
            validation=tf.data.Dataset.from_tensor_slices(validation).batch(self.batch_size, drop_remainder=True),
            epochs=self.n_epochs,
            tensorboard_logs=self.board_path
        )
        logger.write("saving weights...")
        weights = self.model.get_weights()
        du.save_model_obj(os.path.join('weights', str(writer)), weights)
        return weights
