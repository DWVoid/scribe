from typing import *
import numpy as np
import model.utils as du
import matplotlib.pyplot as plt
from utils.logger import LoggerRoot


class Stats:
    stats: Dict[str, List[np.ndarray]] = None

    def __init__(self, args) -> None:
        self.args = args
        self.logger = LoggerRoot(args.log_dir)
        self.all_weights = du.load_model_obj('all_weights', dict())

    @staticmethod
    def transform(raw: Dict[int, List[np.ndarray]]) -> List[np.ndarray]:
        writers: List[int] = sorted(raw.keys())
        count = len(raw[writers[0]])
        results: List[List[np.ndarray]] = [[] for _ in range(count)]
        for writer in writers:
            weights: List[np.ndarray] = raw[writer]
            for i in range(count):
                results[i].append(weights[i])
        return [np.stack(result) for result in results]

    @staticmethod
    def standard_deviation(transformed: List[np.ndarray]) -> List[np.ndarray]:
        return [np.std(item, axis=0) for item in transformed]

    @staticmethod
    def average(transformed: List[np.ndarray]) -> List[np.ndarray]:
        return [np.average(item, axis=0) for item in transformed]

    @staticmethod
    def lower_bound(transformed: List[np.ndarray]) -> List[np.ndarray]:
        return [np.min(item, axis=0) for item in transformed]

    @staticmethod
    def upper_bound(transformed: List[np.ndarray]) -> List[np.ndarray]:
        return [np.max(item, axis=0) for item in transformed]

    def compute(self) -> 'Stats':
        transformed = self.transform(self.all_weights)
        self.stats = {
            'std': self.standard_deviation(transformed),
            'avg': self.average(transformed),
            'min': self.lower_bound(transformed),
            'max': self.upper_bound(transformed)
        }
        return self

    def visualize(self):
        for stat, value in self.stats.items():
            plt.close()
            self.logger.write('visualizing stat: {}'.format(stat))
            for item in value:
                item = item if len(item.shape) > 1 else np.stack([item, item])
                plt.matshow(item)
                plt.colorbar()
            plt.title("stat")
            plt.show()
            input("Press Enter to continue...")

