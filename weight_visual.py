from model import Model
from sample import DataSource
from utils.logger import Logger


def visual_weights(args):
    logger = Logger()  # make logging utility
    logger.write("\nWEIGHT INSPECTION MODE...")
    logger.write("{}\n".format(args))
    logger.write("loading data...")
    data_loader = DataSource(args, logger=logger)
    model = Model(logger=logger)
    logger.write("building model...")