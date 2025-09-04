# from app import app, load_models
from unify import app, run_pipeline
# Load all models before Gunicorn workers start
run_pipeline()

# This is the object Gunicorn imports
application = app
