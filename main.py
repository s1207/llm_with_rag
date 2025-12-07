from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from backend import api
from app import create_dash_app

import flask  # Dash uses Flask internally

# --- Create root FastAPI app ---
app = FastAPI()

# --- Mount FastAPI API sub-app ---
app.mount("/api", api)

# --- Create Flask server for Dash ---
flask_server = flask.Flask(__name__)
dash_app = create_dash_app(flask_server)

# --- Mount Dash frontend at '/' ---
app.mount("/", WSGIMiddleware(flask_server))
