import time
import logging


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.debug('%s %2.2f sec' % (method.__name__, te-ts))
        return result
    return timed