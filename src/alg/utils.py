# Utilities file
import yaml

def load_config(fpath : str) -> dict:
    with open(fpath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg