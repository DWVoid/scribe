import os
import time
import datetime
import threading

max_id = 0
id_lock = threading.Lock()


class Logger:
    __id: int
    __parent: 'Logger'

    def __init__(self, parent):
        global max_id
        global id_lock
        id_lock.acquire()
        self.__id = max_id
        max_id += 1
        id_lock.release()
        self.__parent = parent

    def write(self, s):
        self.__parent.write('[{:>6}]{}'.format(self.__id, s))


class LoggerRoot(Logger):
    def __init__(self, log_dir):
        super().__init__(None)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        path = os.path.join(log_dir, datetime.date.today().isoformat() + '.log')
        self.__sink = open(path, 'a' if not os.path.exists(path) else 'w')
        self.__lock = threading.Lock()
        self.write('Program startup')

    def write(self, s):
        formatted = '[{:>14}]{}'.format(int(time.time()), s)
        self.__lock.acquire()
        print(formatted)
        print(formatted, file=self.__sink, flush=True)
        self.__lock.release()
