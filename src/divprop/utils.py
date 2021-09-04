import copy
import pickle
import logging

from pathlib import Path
from functools import wraps
from hashlib import sha256

DEFAULT_CACHE = Path(".cache/")
DEFAULT_LOGGER = logging.getLogger()


NotFound = object()


def cached_method(method):
    if not hasattr(method, "_cache"):
        method._cache = {}
        assert hasattr(method, "_cache")

    def calc_key(self, *args, **kwargs):
        cls_name = type(self).__name__
        meth_name = method.__name__

        args_key = str((args, sorted(kwargs.items())))
        args_key = sha256(args_key.encode()).hexdigest().upper()
        key = f"{cls_name}.{meth_name}.self_{self.cache_key}.args_{args_key}"
        return key

    @wraps(method)
    def call(self, *args, **kwargs):
        log = getattr(self, "log", DEFAULT_LOGGER)
        CACHE = getattr(self, "CACHE", DEFAULT_CACHE)
        if CACHE:
            self.CACHE = Path(CACHE)

            if not CACHE.is_dir():
                log.debug(f"cache folder {self.CACHE} does not exist, disabling cache")
                self.CACHE = None
        else:
            log.debug("cache disabled")
            self.CACHE = None

        key = calc_key(self, *args, **kwargs)

        ret = method._cache.get(key, NotFound)
        if ret is not NotFound:
            return copy.deepcopy(ret)

        if self.CACHE:
            cache_filename = self.CACHE / key
            try:
                ret = pickle.load(open(cache_filename, "rb"))
                log.info(f"load {cache_filename} succeeded")
            except Exception as err:
                log.debug(f"load {cache_filename} failed: {err}")
                pass

            if ret is not NotFound:
                method._cache[key] = ret
                return copy.deepcopy(ret)

        log.info(f"computing {key}")

        ret = method(self, *args, **kwargs)
        method._cache[key] = ret

        if self.CACHE:
            pickle.dump(ret, open(cache_filename, "wb"))
            log.info(f"save {cache_filename} succeeded")
        return copy.deepcopy(ret)

        call.calc_key = calc_key
    return call


def cached_func(method=None, CACHE=DEFAULT_CACHE, log=DEFAULT_LOGGER):
    def deco(method):
        nonlocal CACHE, log
        method._cache = {}
        if CACHE:
            CACHE = Path(CACHE)
            if not CACHE.is_dir():
                log.warning(f"cache folder {CACHE} does not exist, disabling cache")
                CACHE = None
        else:
            log.warning("cache disabled")
            CACHE = None

        @wraps(method)
        def call(*args, **kwargs):
            meth_name = method.__name__

            args_key = str((args, sorted(kwargs.items())))
            args_key = sha256(args_key.encode()).hexdigest().upper()
            key = f"{meth_name}.args_{args_key}"

            ret = method._cache.get(key, NotFound)
            if ret is not NotFound:
                return copy.deepcopy(ret)

            if CACHE:
                cache_filename = CACHE / key
                try:
                    ret = pickle.load(open(cache_filename, "rb"))
                    log.info(f"load {cache_filename} succeeded")
                except Exception as err:
                    log.warning(f"load {cache_filename} failed: {err}")
                    pass

                if ret is not NotFound:
                    method._cache[key] = ret
                    return copy.deepcopy(ret)

            log.info(f"computing {key}")

            ret = method(*args, **kwargs)
            method._cache[key] = ret

            if CACHE:
                pickle.dump(ret, open(cache_filename, "wb"))
                log.info(f"save {cache_filename} succeeded")
            return copy.deepcopy(ret)
        return call
    if method is None:
        return deco
    return deco(method)
