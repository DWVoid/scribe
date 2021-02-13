import os
import shutil
import model.utils as du

from model.compiler import DataSetCompiler
from model.parser import IAMParser
from model.types import *
from utils.logger import Logger


class DataSource:
    compiled: DataSetsCompiled
    logger: Logger

    def __init__(self, args, logger: Logger, limit=500) -> None:
        self.logger = logger
        # alias some variables for easy referencing
        meta = DataSetMeta()
        meta.transfer(args, limit)
        if self.__try_load_compiled(meta):
            return
        if self.__try_compile_parsed(meta):
            return
        if self.__try_parse_compile(meta):
            return

    def datasets(self):
        return self.compiled.items()

    def __try_load_compiled(self, meta: DataSetMeta) -> bool:
        meta_cached: DataSetMeta = du.load_cache('compiled.meta')
        if meta_cached is None:
            return False  # nothing has been cached
        if not (meta == meta_cached):
            return False  # the parameters for compilation has changed
        self.compiled: DataSetsCompiled = du.load_cache('compiled')
        return self.compiled is not None

    def __try_compile_parsed(self, meta: DataSetMeta) -> bool:
        parsed: DataSetsRaw = du.load_cache('parsed')
        if parsed is None:
            return False  # parsed file does not exist
        else:
            self.__compile(meta, parsed)
            return True

    def __compile(self, meta: DataSetMeta, parsed: DataSetsRaw) -> None:
        result: DataSetsCompiled = dict()
        compiler = DataSetCompiler(meta, Logger(self.logger))
        for writer, raw in parsed.items():
            result[writer] = compiler.compile(raw)
        du.save_cache('compiled', result)
        du.save_cache('compiled.meta', meta)
        self.compiled = result

    def __try_parse_compile(self, meta: DataSetMeta) -> bool:
        ascii_path = du.cache_path('ascii')
        strokes_path = du.cache_path('lineStrokes')
        ascii_tar_path = du.data_path('ascii-all.tar.gz')
        strokes_tar_path = du.data_path('lineStrokes-all.tar.gz')
        forms_path = du.data_path('forms.txt')
        if not (os.path.exists(strokes_tar_path) and os.path.exists(ascii_tar_path) and os.path.exists(forms_path)):
            return False  # data partially missing
        # unpack the tar data if that is not there
        if not os.path.exists(strokes_path):
            self.logger.write('unpacking strokes data...')
            shutil.unpack_archive(strokes_tar_path, extract_dir=du.cache_path())
        if not os.path.exists(ascii_path):
            self.logger.write('unpacking ascii data...')
            shutil.unpack_archive(ascii_tar_path, extract_dir=du.cache_path())
        # parse and save the raw Set
        raw_set = IAMParser(Logger(self.logger)).run(strokes_path, ascii_path, forms_path)
        du.save_cache('parsed', raw_set)
        # call the Set compiler
        self.__compile(meta, raw_set)
        return True
