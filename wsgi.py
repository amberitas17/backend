from app import app, load_models

# Load all models before Gunicorn workers start
load_models()

# This is the object Gunicorn imports
application = app
