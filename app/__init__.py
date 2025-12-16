from flask import Flask, request, g
import time
import uuid

from .routes import api
from .logging_config import setup_logging


def create_app():
    app = Flask(__name__)
    setup_logging(app)

    @app.before_request
    def _start_timer():
        g.request_id = uuid.uuid4().hex[:10]
        g.t0 = time.time()

    @app.after_request
    def _log_request(resp):
        dt_ms = int((time.time() - g.t0) * 1000)
        app.logger.info(f"[{g.request_id}] {request.method} {request.path} -> {resp.status_code} ({dt_ms} ms)")
        return resp

    app.register_blueprint(api)
    return app
