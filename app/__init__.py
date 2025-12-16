from flask import Flask, request, g
import time
import uuid

from .routes import api
from .logging_config import setup_logging
from .log_context import request_id_var, run_id_var

def create_app():
    app = Flask(__name__)
    setup_logging(app)

    @app.before_request
    def _before():
        g._t0 = time.time()
        rid = uuid.uuid4().hex[:10]
        g._rid_tok = request_id_var.set(rid)
        g._run_tok = run_id_var.set("-")  # default unless pipeline binds a run_id

        app.logger.debug(f"REQUEST_START method={request.method} path={request.path} args={dict(request.args)}")

    @app.after_request
    def _after(resp):
        dt_ms = int((time.time() - g._t0) * 1000)
        app.logger.info(f"REQUEST_END {request.method} {request.path} status={resp.status_code} dt_ms={dt_ms}")
        return resp

    @app.teardown_request
    def _teardown(exc):
        # ensure contextvars are reset even on exceptions
        try:
            request_id_var.reset(getattr(g, "_rid_tok", None))
        except Exception:
            pass
        try:
            run_id_var.reset(getattr(g, "_run_tok", None))
        except Exception:
            pass
        if exc:
            app.logger.exception(f"TEARDOWN_EXCEPTION: {exc}")

    app.register_blueprint(api)
    return app
