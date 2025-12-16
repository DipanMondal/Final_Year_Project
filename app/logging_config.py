import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(app):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    level = logging.INFO
    app.logger.setLevel(level)

    # avoid double handlers on reload
    if any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)

    app.logger.addHandler(fh)
    app.logger.addHandler(sh)

    app.logger.info("Logging initialized")
