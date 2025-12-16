from contextvars import ContextVar
from contextlib import contextmanager

request_id_var = ContextVar("request_id", default="-")
run_id_var = ContextVar("run_id", default="-")

@contextmanager
def bind_run_id(run_id: str):
    tok = run_id_var.set(run_id)
    try:
        yield
    finally:
        run_id_var.reset(tok)
