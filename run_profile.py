# run_profile.py

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine

engine = ProfileEngine(load_default_config())
engine.run()
