from typing import *

import os
import shutil
import numpy as np
import pickle as pickle
import xml.etree.ElementTree as ET


# abstraction for logging
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


# stores the meta-data of the compiled Set. if this changes, the Set needs to be re-compiled
class DataSetMeta:
    limit: int
    t_steps: int
    alphabet: str
    data_scale: int
    ascii_steps: int
    sample_space_size: int

    def transfer(self, args, limit: int) -> None:
        self.limit = limit  # removes large noisy gaps in the data
        self.t_steps = args.tsteps
        self.alphabet = args.alphabet
        self.data_scale = args.data_scale  # scale data down by this factor
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)
        self.sample_space_size = args.batch_size * args.nbatches

    def __eq__(self, other) -> bool:
        return self.limit == other.limit \
               and self.t_steps == other.t_steps \
               and self.alphabet == other.alphabet \
               and self.data_scale == other.data_scale \
               and self.ascii_steps == other.ascii_steps \
               and self.sample_space_size == other.sample_space_size


DataSetRaw = List[Tuple[np.ndarray, str]]
DataSetShaped = Tuple[Dict[str, List[np.ndarray]], List[np.ndarray]]
DataSetCompiled = Tuple[DataSetShaped, DataSetShaped]


# a parsed Set. the same for a given database
class DataSetsRaw:
    writers: List[int]
    datasets: Dict[int, DataSetRaw]


# a compiled Set. the data-structure is the same used to train the model based on the given meta parameters
class DataSetsCompiled:
    writers: List[int]
    datasets: Dict[int, DataSetCompiled]


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


# parser for IAM online handwriting database
class IAMParser:
    text_data: Dict[str, List[str]] = dict()
    stroke_data: Dict[str, np.ndarray] = dict()
    writer_data: Dict[str, int] = dict()
    logger: Logger

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def run(self, stroke_dir: str, ascii_dir: str, writer_file: str) -> Dict[int, DataSetRaw]:
        result: Dict[int, DataSetRaw] = dict()
        self.logger.write("\tparsing dataset...")
        self.__load_writers(writer_file)
        self.__load_all_text(ascii_dir)
        self.__load_all_strokes(stroke_dir)
        self.logger.write("\t\tcomputing, please wait...")
        for name, strokes in self.stroke_data.items():
            session_id = name[:-3]
            # find the corresponding text line
            text_session = self.text_data.get(session_id)
            if text_session is None:
                self.logger.write("\t\tno transcript for stoke: " + name)
                continue
            line_number = int(name[-2:]) - 1
            if len(text_session) <= line_number:
                self.logger.write("\t\tno line transcript for stoke: " + name)
                continue
            text = text_session[line_number]
            # find the writer
            writer = self.writer_data.get(session_id)
            if writer is None:
                self.logger.write("\t\tno writer for stoke: {} text: {}".format(name, text))
                continue
            array = result.get(writer, [])
            array.append((strokes, text))
            result[writer] = array
        self.logger.write("\tfinished parsing Set")
        return result

    def __load_all_strokes(self, directory: str) -> None:
        self.logger.write("\t\tloading strokes...")
        for file in self.__list_files(directory, 'xml'):
            self.stroke_data[os.path.basename(file)[:-4]] = self.__stroke_to_array(self.__load_strokes(file))

    def __load_all_text(self, directory: str) -> None:
        self.logger.write("\t\tloading text...")
        for file in self.__list_files(directory, 'txt'):
            self.text_data[os.path.basename(file)[:-4]] = self.__load_text(file)

    # read a writer file and return a dictionary from form id to the writer id
    def __load_writers(self, writer_file: str) -> None:
        self.logger.write("\t\tloading writers...")
        with open(writer_file, 'r') as f:
            contents = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            contents = [x.strip().split(" ") for x in contents]
            self.writer_data = {content[0]: int(content[1]) for content in contents}

    @staticmethod
    def __list_files(root_dir: str, extension: str) -> List[str]:
        return [os.path.join(dir_name, name)
                for dir_name, subdir_list, current_list in os.walk(root_dir)
                for name in current_list if name[-3:] == extension]

    # function to read each individual xml file
    @staticmethod
    def __load_strokes(filename: str) -> List[List[List[int]]]:
        root = ET.parse(filename).getroot()
        # get rid of side spaces
        x = y = 1e20
        height = 0
        for i in range(1, 4):
            x = min(x, float(root[0][i].attrib['x']))
            y = min(y, float(root[0][i].attrib['y']))
            height = max(height, float(root[0][i].attrib['y']))
        height -= y
        x -= 100
        y -= 100
        return [[[float(p.attrib['x']) - x, float(p.attrib['y']) - y] for p in stroke.findall('Point')]
                for stroke in root[1].findall('Stroke')]

    # converts a list of arrays into a 2d numpy int16 array
    @staticmethod
    def __stroke_to_array(stroke: List[List[List[int]]]) -> np.ndarray:
        n_point = 0
        for i in range(len(stroke)):
            n_point += len(stroke[i])
        stroke_data = np.zeros((n_point, 3), dtype=np.int16)
        prev_x = prev_y = counter = 0
        for j in range(len(stroke)):
            for k in range(len(stroke[j])):
                stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                prev_x = int(stroke[j][k][0])
                prev_y = int(stroke[j][k][1])
                stroke_data[counter, 2] = 1 if k == (len(stroke[j]) - 1) else 0  # 1: end of stroke
                counter += 1
        return stroke_data

    @staticmethod
    def __load_text(file: str) -> List[str]:
        with open(file, 'r') as f:
            s = f.read()
            return s[s.find('CSR'):].split('\n')[2:]


class DataSource:
    compiled: DataSetsCompiled
    logger: Logger

    def __init__(self, args, logger: Logger, limit=500) -> None:
        self.logger = logger
        # alias some variables for easy referencing
        data_dir = args.data_dir
        cache_dir = args.cache_dir
        # create the cache dir if it is missing
        if not (os.path.exists(cache_dir)):
            os.mkdir(cache_dir)
        meta = DataSetMeta()
        meta.transfer(args, limit)
        if self.__try_load_compiled(meta, cache_dir):
            return
        if self.__try_compile_parsed(meta, cache_dir):
            return
        if self.__try_parse_compile(meta, cache_dir, data_dir):
            return

    def datasets(self):
        return self.compiled.datasets.items()

    def __try_load_compiled(self, meta: DataSetMeta, cache_dir: str) -> bool:
        meta_name: AnyStr = os.path.join(cache_dir, 'compiled.meta.pkl')
        if not os.path.exists(meta_name):
            return False  # nothing has been cached
        meta_cached = self.__pickle_load(meta_name)
        if not (meta == meta_cached):
            return False  # the parameters for compilation has changed
        compiled_name: AnyStr = os.path.join(cache_dir, 'compiled.pkl')
        if not os.path.exists(compiled_name):
            return False  # for some reason the cached file is missing
        self.compiled = self.__pickle_load(compiled_name)
        return True

    def __try_compile_parsed(self, meta: DataSetMeta, cache_dir: str) -> bool:
        parsed_name: AnyStr = os.path.join(cache_dir, 'parsed.pkl')
        if not os.path.exists(parsed_name):
            return False  # parsed file does not exist
        self.__compile(meta, self.__pickle_load(parsed_name), cache_dir)
        return True

    def __compile(self, meta: DataSetMeta, parsed: DataSetsRaw, cache_dir: str) -> None:
        result = DataSetsCompiled()
        result.writers = parsed.writers
        result.datasets = dict()
        compiler = DataSetCompiler(meta, self.logger)
        for writer, raw in parsed.datasets.items():
            result.datasets[writer] = compiler.compile(raw)
        self.__pickle_save(os.path.join(cache_dir, 'compiled.pkl'), result)
        self.__pickle_save(os.path.join(cache_dir, 'compiled.meta.pkl'), meta)
        self.compiled = result

    def __try_parse_compile(self, meta: DataSetMeta, cache_dir: str, data_dir: str) -> bool:
        if not os.path.exists(data_dir):
            return False  # data not present
        ascii_path: AnyStr = os.path.join(cache_dir, 'ascii')
        strokes_path: AnyStr = os.path.join(cache_dir, 'lineStrokes')
        ascii_tar_path: AnyStr = os.path.join(data_dir, 'ascii-all.tar.gz')
        strokes_tar_path: AnyStr = os.path.join(data_dir, 'lineStrokes-all.tar.gz')
        forms_path: AnyStr = os.path.join(data_dir, 'forms.txt')
        if not (os.path.exists(strokes_tar_path) and os.path.exists(ascii_tar_path) and os.path.exists(forms_path)):
            return False  # data partially missing
        # unpack the tar data if that is not there
        if not os.path.exists(strokes_path):
            self.logger.write('\t\tunpacking strokes data...')
            shutil.unpack_archive(strokes_tar_path, extract_dir=cache_dir)
        if not os.path.exists(ascii_path):
            self.logger.write('\t\tunpacking ascii data...')
            shutil.unpack_archive(ascii_tar_path, extract_dir=cache_dir)
        # parse and save the raw Set
        parser = IAMParser(self.logger)
        parsed = parser.run(strokes_path, ascii_path, forms_path)
        raw_set = DataSetsRaw()
        raw_set.writers = list(parsed.keys())
        raw_set.datasets = parsed
        self.__pickle_save(os.path.join(cache_dir, 'parsed.pkl'), raw_set)
        # call the Set compiler
        self.__compile(meta, raw_set, cache_dir)
        return True

    @staticmethod
    def __pickle_load(file: AnyStr) -> object:  # I have no idea how to type this
        with open(file, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def __pickle_save(file: AnyStr, obj: object) -> None:
        with open(file, 'wb') as f:
            pickle.dump(obj, f, protocol=2)
