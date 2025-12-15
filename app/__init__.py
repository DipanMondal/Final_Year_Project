from flask import Flask
from .routes import api, init_services

def create_app():
    app = Flask(__name__)

    # initialize model/services once at startup
    with app.app_context():
        init_services()

    app.register_blueprint(api)
    return app
