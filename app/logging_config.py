import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .log_context import request_id_var, run_id_var

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.run_id = run_id_var.get()
        return True

def setup_logging(app):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    app.logger.setLevel(level)

    # Avoid double handlers (Flask reloader)
    if getattr(app, "_logging_setup_done", False):
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | rid=%(request_id)s | run=%(run_id)s | %(name)s | %(message)s"
    )

    fh = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=3_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    fh.addFilter(ContextFilter())

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    sh.addFilter(ContextFilter())

    app.logger.addHandler(fh)
    app.logger.addHandler(sh)

    # Also configure root logger so modules using logging.getLogger(__name__) work
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)

    app._logging_setup_done = True
    app.logger.info(f"Logging initialized (level={level_name})")
