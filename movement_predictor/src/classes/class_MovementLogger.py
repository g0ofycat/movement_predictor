import json
import time
import keyboard
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from src.config.settings import settings

class MovementLogger:
    def __init__(self, stop_key: str, save_path: str, valid_keys: list[str]):
        self.collecting = False
        self.data: List[Dict] = []
        self.last_action = None
        self.last_action_time = None
        self.valid_keys = valid_keys
        self.stop_key = stop_key.lower()
        self.save_path = save_path

    def start_logging(self):
        self.collecting = True
        self.data.clear()
        print(f"Starting key logging. Press {self.stop_key.upper()} to stop")
        self._start_keyboard_detection()

    def stop_logging(self):
        self.collecting = False
        keyboard.unhook_all()
        self.save_json()

    def save_json(self):
        if not self.data:
            print("No data collected to save")
            return
        try:
            X, Y = self._convert_to_training_format()
            X_list, Y_list = X.tolist(), Y.tolist()  

            save_dict = {
                'raw_data': self.data,
                'training_data': {
                    'X': X_list,
                    'Y': Y_list
                }
            }

            with open(self.save_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

            print(f"Data saved to: {self.save_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _start_keyboard_detection(self):
        def on_key_event(event):
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name.lower()
                if key in self.valid_keys and self.collecting:
                    self._log_action(key.lower())

        keyboard.hook(on_key_event)

        try:
            keyboard.wait(self.stop_key)
        except KeyboardInterrupt:
            print(f"\nLogging stopped by '{self.stop_key}'")
        finally:
            self.stop_logging()

    def _log_action(self, action: str):
        now = time.time()
        delta = now - self.last_action_time if self.last_action_time else 0
        max_delta = 10.0
        normalized_delta = min(delta / max_delta, 1.0)

        self.data.append({
            'timestamp': now,
            'datetime': str(datetime.fromtimestamp(now)),
            'action': action,
            'time_since_last_action': normalized_delta,
            'previous_action': self.last_action,
            'action_sequence': [d['action'] for d in self.data[-5:]]
        })

        self.last_action, self.last_action_time = action, now
        print(f"[{len(self.data)}] {action.upper()} (Time since last: {delta:.2f}s, Normalized Time: {normalized_delta:.2f})")

    def _convert_to_training_format(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.data:
            return np.array([]), np.array([])

        X, Y = [], []

        action_map = {k: [1 if i == idx else 0 for i in range(len(settings['settings']['valid_keys']))] for idx, k in enumerate(settings['settings']['valid_keys'])}

        for d in self.data[1:]:
            prev = d['previous_action']
            
            X.append([d['time_since_last_action']] + [1 if prev == k else 0 for k in settings['settings']['valid_keys']])
            Y.append(action_map[d['action']])

        return np.array(X), np.array(Y)
    
    @classmethod
    def load_training_data(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        X = np.array(data['training_data']['X'])
        Y = np.array(data['training_data']['Y'])
        
        return X, Y