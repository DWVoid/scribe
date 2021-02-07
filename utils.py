import numpy as np
import os
import pickle as pickle
import xml.etree.ElementTree as ET


# create data file from raw xml files from iam handwriting source.
# noinspection PyMethodMayBeStatic
class DataParser:
    def __init__(self, args, logger, limit):
        self.logger = logger
        self.text_temp = {}
        self.tsteps = args.tsteps
        self.alphabet = args.alphabet
        self.data_scale = args.data_scale  # scale data down by this factor
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)
        self.limit = limit # removes large noisy gaps in the data

    def run(self, stroke_dir, ascii_dir, writer_file, data_file):
        self.logger.write("\tparsing dataset...")
        form_to_writer = self.__load_writer_file(writer_file)
        writer_to_data = {}

        # for each line of words in the dir
        for stroke_file in self.__list_xml_files(stroke_dir):
            ascii_file = stroke_file.replace(stroke_dir, ascii_dir)[:-7] + '.txt'
            text_ascii = self.__load_ascii_file(ascii_file, int(stroke_file[-6:-4]) - 1)
            # check if length of the text is less than 6
            if len(text_ascii) < self.ascii_steps:
                self.logger.write("\tline length was too short. line was: " + text_ascii)
                continue
            # get the writer id
            writer_id = form_to_writer.get(ascii_file.split("/")[-1][:-4], "")
            # check if the writer id is valid
            if writer_id == "":
                self.logger.write("\tno writer id. line was: " + text_ascii)
                continue
            # get the stroke array from the xml file
            stroke_array = self.__stroke_to_array(self.__load_stroke_file(stroke_file))
            # check if the length of the stroke array is less than 150
            if len(stroke_array) < self.tsteps + 1:
                self.logger.write("\tlength of stroke points is less than 150. line was: " + text_ascii)
                continue
            data = self.__preprocess_data(stroke_array, text_ascii)
            # add the strokes and asciis under the specific writer
            # print(writer_id)
            # print(writer_to_data.get(writer_id, []), "\n")
            writer_to_data[writer_id] = writer_to_data.get(writer_id, []) + [data]

        with open(data_file, 'wb') as f:
            pickle.dump(writer_to_data, f, protocol=2)
        self.logger.write("\tfinished parsing dataset")

    # preprocess data to fit the input of the model
    def __preprocess_data(self, stroke_array, text_ascii):
        # removes large gaps from the data and convert to float32
        stroke_array = np.array(np.clip(stroke_array, -self.limit, self.limit), dtype=np.float32)
        # scale the data down
        stroke_array[:, 0:2] /= self.data_scale
        # put the data in the form of input
        t_x = stroke_array[:self.tsteps]
        t_y = stroke_array[1:self.tsteps + 1]
        t_c = to_one_hot(text_ascii, self.ascii_steps, self.alphabet)
        return ({'stroke': t_x, 'char': t_c}, t_y)

    # read a writer file and return a dictionary from form id to the writer id
    def __load_writer_file(self, writer_file):
        with open(writer_file, 'r') as f:
            contents = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            contents = [x.strip().split(" ") for x in contents]
            form_to_writer = {content[0]: int(content[1]) for content in contents}

        return form_to_writer

    def __list_xml_files(self, root_dir):
        return [os.path.join(dir_name, name)
                for dir_name, subdir_list, current_list in os.walk(root_dir)
                for name in current_list if name[-3:] == 'xml']

    # function to read each individual xml file
    def __load_stroke_file(self, filename):
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

    def __load_ascii_file(self, file, line):
        if file not in self.text_temp:
            with open(file, 'r') as f:
                s = f.read()
            self.text_temp[file] = lines = s[s.find('CSR'):].split('\n')
        else:
            lines = self.text_temp[file]
        return lines[line + 2] if len(lines) > line + 2 else ''

    # converts a list of arrays into a 2d numpy int16 array
    def __stroke_to_array(self, stroke):
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

class DataLoader:
    def __init__(self, args, logger, limit=500):
        self.batch_size = args.batch_size
        self.logger = logger
        self.limit = limit  # removes large noisy gaps in the data

        data_dir = args.data_dir
        data_file = os.path.join(data_dir, "strokes_training_data.cpkl")

        if not (os.path.exists(data_file)):
            self.logger.write("\tcreating training data cpkl file from raw source")
            DataParser(args, logger, limit).run(os.path.join(data_dir, 'lineStrokes'), os.path.join(data_dir, 'ascii'),
                                   os.path.join(data_dir, 'forms.txt'), data_file)

        self.logger.write("\tloading dataset...")
        with open(data_file, 'rb') as f:
           writer_to_data = pickle.load(f)
        
        self.logger.write("\tassembling dataset...")
        self.writers = list(writer_to_data.keys())
        self.writer_to_manager = {writer_id : WriterDataManager(data, self.batch_size, logger) for writer_id, data in writer_to_data.items()}
        self.logger.write("\t\t{} writers loaded".format(len(self.writers)))
    
    def validation_data(self, writer_id):
        return self.writer_to_manager[writer_id].validation_data()

    def training_data(self, writer_id, batches):
        return self.writer_to_manager[writer_id].training_data(batches)

# keep the data of a single writer
class WriterDataManager:
    def __init__(self, data, batch_size, logger):
        self.batch_size = batch_size
        self.logger = logger

        self.__load_preprocessed(data)
        self.reset_batch_pointer()

    def __load_preprocessed(self, data):
        # goes thru the list, and only keeps the text entries that have more than tsteps points
        # every 1 in 20 (5%) will be used for validation data
        
        self.train_data = []
        self.valid_data = []
        cur_data_counter = 0
        for i, data in enumerate(data):
            if cur_data_counter % 20 == 0:
                self.valid_data.append(data)
            else:
                self.train_data.append(data)
            cur_data_counter += 1

        # minus 1, since we want the ydata to be a shifted version of x data
        # self.num_batches = int(len(self.stroke_data) / self.batch_size)
        # self.logger.write("\t\t{} train individual data points".format(len(self.stroke_data)))
        # self.logger.write("\t\t{} valid individual data points".format(len(self.valid_stroke_data)))
        # self.logger.write("\t\t{} batches".format(self.num_batches))

    # returns validation data
    def validation_data(self):
        data_batch = []
        for i in range(self.batch_size):
            index = i % len(self.valid_stroke_data)
            data_batch .append(self.valid_stroke_data[index])
        return data_batch

    # returns (batches) randomized, tsteps-sized portion of the training data
    def training_data(self, batches):
        data_batch = []
        for _ in range(batches):
            for i in range(self.batch_size):
                index = self.idx_perm[self.pointer]
                data_batch.append(self.train_data[index])
                self.tick_batch_pointer()
        return data_batch

    def tick_batch_pointer(self):
        self.pointer += 1
        if self.pointer >= len(self.train_data):
            self.reset_batch_pointer()

    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.train_data))
        self.pointer = 0


# utility function for converting input ascii characters into vectors the network can understand.
# index position 0 means "unknown"
def to_one_hot(s, ascii_steps, alphabet):
    s = s[:3e3] if len(s) > 3e3 else s  # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        seq = seq[:ascii_steps]
    else:
        seq = seq + [0] * (ascii_steps - len(seq))
    one_hot = np.zeros((ascii_steps, len(alphabet) + 1))
    one_hot[np.arange(ascii_steps), seq] = 1
    return one_hot


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
