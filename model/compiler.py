from model.types import *
from utils.logger import Logger


# class for compiling the Set of a single writer
class DataSetCompiler:
    meta: DataSetMeta
    logger: Logger

    def __init__(self, meta: DataSetMeta, logger: Logger) -> None:
        self.meta = meta
        self.logger = logger

    def compile(self, raw: DataSetRaw) -> DataSetCompiled:
        # goes thru the list, and only keeps the text entries that have more than tsteps points
        train_data_raw, valid_data_raw = self.__split_data(raw)
        train_data_padded = self.__pad_train_data(train_data_raw)
        return self.__transform(train_data_padded), self.__transform(valid_data_raw)

    def __split_data(self, raw: DataSetRaw) -> Tuple[DataSetRaw, DataSetRaw]:
        train_data: DataSetRaw = list()
        valid_data: DataSetRaw = list()
        cur_data_counter = 0
        for data in raw:
            strokes, text = data
            # check for data points that is shorter than the step sizes set and drop those
            if len(text) < self.meta.ascii_steps:
                self.logger.write("\tline length was too short. line was: " + text)
                continue
            if len(strokes) < self.meta.t_steps + 1:
                self.logger.write("\tnot enough stroke points. count: {} line: {}".format(len(strokes), text))
                continue
            # every 1 in 20 (5%) will be reserved for validation data
            if cur_data_counter % 20 == 0:
                valid_data.append(data)
            else:
                train_data.append(data)
            cur_data_counter += 1
        return train_data, valid_data

    def __transform(self, raw: DataSetRaw) -> DataSetShaped:
        x: List[np.ndarray] = list()
        y: List[np.ndarray] = list()
        c: List[np.ndarray] = list()
        for stroke_raw, text in raw:
            # removes large gaps from the data and convert to float32
            stroke_array = np.array(np.clip(stroke_raw, -self.meta.limit, self.meta.limit), dtype=np.float32)
            # scale the data down
            stroke_array[:, 0:2] /= self.meta.data_scale
            # put the data in the form of input
            x.append(stroke_array[:self.meta.t_steps])
            y.append(stroke_array[1:self.meta.t_steps + 1])
            c.append(self.__encode_one_hot(text))
        return {'stroke': x, 'char': c}, y

    def __pad_train_data(self, raw: DataSetRaw) -> DataSetRaw:
        pointer = 0
        perm = np.random.permutation(len(raw))
        result: DataSetRaw = list()
        for _ in range(self.meta.sample_space_size):
            if pointer == len(perm):
                pointer = 0
                perm = np.random.permutation(len(perm))
            result.append(raw[perm[pointer]])
            pointer += 1
        return result

    # index position 0 means "unknown"
    def __encode_one_hot(self, s: str) -> np.ndarray:
        s = s[:3e3] if len(s) > 3e3 else s  # clip super-long strings
        seq = [self.meta.alphabet.find(char) + 1 for char in s]
        if len(seq) >= self.meta.ascii_steps:
            seq = seq[:self.meta.ascii_steps]
        else:
            seq = seq + [0] * (self.meta.ascii_steps - len(seq))
        one_hot = np.zeros((self.meta.ascii_steps, len(self.meta.alphabet) + 1), dtype=np.float32)
        one_hot[np.arange(self.meta.ascii_steps), seq] = 1
        return one_hot