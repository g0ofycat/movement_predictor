import json
import os
from pathlib import Path

def load_settings():
    settings_dir = Path(__file__).parent
    project_root = settings_dir.parent.parent.parent
    
    settings_file = settings_dir / 'settings_data.json'
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    info_path = settings['settings']['information']
    if not os.path.isabs(info_path):
        info_path = project_root / info_path
    
    with open(info_path, 'r') as f:
        info_data = json.load(f)
    
    settings.update({
        'information': info_data.get('information', {}),
        'keyboard_settings': info_data.get('keyboard_settings', {})
    })
    
    return settings

settings = load_settings()