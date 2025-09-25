import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# === IMPORTS ===

from src.training.class_MovementLogger import MovementLogger
from src.config.settings import settings

# === LOGGING ===

Logger = MovementLogger(stop_key='q', save_path=settings['settings']['movement_data_path'], valid_keys=settings['keyboard_settings']['valid_keys'])

Logger.start_logging()