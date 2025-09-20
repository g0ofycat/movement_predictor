import json
import os

def load_settings():
    settings_dir = os.path.dirname(__file__)
    
    with open(os.path.join(settings_dir, 'settings_data.json'), 'r') as f:
        settings = json.load(f)
    
    return settings

settings = load_settings()