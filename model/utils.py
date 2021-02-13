from typing import *
import pickle
import os

data_dir: AnyStr
cache_dir: AnyStr
model_obj_dir: AnyStr


def set_path(
        base: AnyStr,
        data: AnyStr = 'data',
        cache: AnyStr = 'cache',
        model_obj: AnyStr = 'saved'
):
    global data_dir
    global cache_dir
    global model_obj_dir
    data_dir = os.path.join(base, data)
    cache_dir = os.path.join(base, cache)
    model_obj_dir = os.path.join(base, model_obj)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if not os.path.exists(model_obj_dir):
        os.mkdir(model_obj_dir)


def load_p2(file: AnyStr, default: Any) -> Any:  # I have no idea how to type this
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        return default


def save_p2(file: AnyStr, obj: Any) -> None:
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def cache_path(sub: AnyStr = '') -> AnyStr:
    return os.path.join(cache_dir, sub)


def data_path(sub: AnyStr = '') -> AnyStr:
    return os.path.join(data_dir, sub)


def model_obj_path(sub: AnyStr = '') -> AnyStr:
    return os.path.join(model_obj_dir, sub)


def load_cache(name: AnyStr, default: Any = None) -> Any:
    return load_p2(cache_path(name + '.pkl2'), default)


def save_cache(name: AnyStr, obj: Any) -> None:
    return save_p2(cache_path(name + '.pkl2'), obj)


def load_data(name: AnyStr, default: Any = None) -> Any:
    return load_p2(data_path(name + '.pkl2'), default)


def save_data(name: AnyStr, obj: Any) -> None:
    return save_p2(data_path(name + '.pkl2'), obj)


def load_model_obj(name: AnyStr, default: Any = None) -> Any:
    return load_p2(model_obj_path(name + '.pkl2'), default)


def save_model_obj(name: AnyStr, obj: Any) -> None:
    return save_p2(model_obj_path(name + '.pkl2'), obj)
