import json
import os
from pathlib import Path

def load_settings():
    settings_dir = Path(__file__).parent
    project_root = settings_dir.parent.parent.parent
    
    settings_file = settings_dir / 'settings_data.json'
    
    with open(settings_file, 'r') as f:
        settings = json.load(f)

    model_name = settings['settings']['model_name']
    model_version = settings['settings']['model_version']

    settings['settings']['movement_data_path'] = settings['settings']['movement_data_path'].format(
        model_name=model_name, 
        model_version=model_version
    )
    settings['settings']['model_data_path'] = settings['settings']['model_data_path'].format(
        model_name=model_name, 
        model_version=model_version
    )
    settings['settings']['information'] = settings['settings']['information'].format(
        model_name=model_name
    )

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