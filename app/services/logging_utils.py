import time
import logging
import functools

def trace(_fn=None, *, level=logging.DEBUG, log_args=False):
    def deco(fn):
        logger = logging.getLogger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            if log_args:
                logger.log(level, f"ENTER {fn.__name__} args={args} kwargs={kwargs}")
            else:
                logger.log(level, f"ENTER {fn.__name__}")
            try:
                out = fn(*args, **kwargs)
                dt = (time.time() - t0) * 1000
                logger.log(level, f"EXIT {fn.__name__} dt_ms={dt:.1f}")
                return out
            except Exception as e:
                dt = (time.time() - t0) * 1000
                logger.exception(f"EXCEPTION {fn.__name__} dt_ms={dt:.1f} err={e}")
                raise

        return wrapper

    if _fn is None:
        return deco
    return deco(_fn)
