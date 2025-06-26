# run_profile.py
import yaml

from dsbf.eda.profile_engine import ProfileEngine

with open("example_config.yaml", "r") as f:
    config = yaml.safe_load(f)

engine = ProfileEngine(config)
engine.run()
