import os
from model.types import *
from utils.logger import Logger
from xml.etree import ElementTree as ETree


# parser for IAM online handwriting database
class IAMParser:
    text_data: Dict[str, List[str]] = dict()
    stroke_data: Dict[str, np.ndarray] = dict()
    writer_data: Dict[str, int] = dict()
    logger: Logger

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def run(self, stroke_dir: str, ascii_dir: str, writer_file: str) -> DataSetsRaw:
        result: DataSetsRaw = dict()
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
        root = ETree.parse(filename).getroot()
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