import json
import time
import keyboard
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class MovementLogger:
    def __init__(self, stop_key: str, save_path: str, valid_keys: List[str]):
        self.collecting = False
        self.data: List[Dict] = []
        self.last_action_time = None
        self.valid_keys = valid_keys
        self.stop_key = stop_key.lower()
        self.save_path = save_path
        self.key_press_times = {}
        self.currently_pressed = set()

    def start_logging(self):
        self.collecting = True
        self.data.clear()
        print(f"Starting key logging. Press '{self.stop_key.upper()}' to stop")
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
            
            save_dict = {
                'raw_data': self.data,
                'training_data': {
                    'X': X.tolist(),
                    'Y': Y.tolist()
                },
                'summary': self._get_summary()
            }

            with open(self.save_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

            print(f"Data saved to: {self.save_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _start_keyboard_detection(self):
        def on_key_event(event):
            key = event.name.lower()
            if key in self.valid_keys and self.collecting:
                if event.event_type == keyboard.KEY_DOWN and key not in self.currently_pressed:
                    self._log_key_down(key)
                elif event.event_type == keyboard.KEY_UP and key in self.currently_pressed:
                    self._log_key_up(key)

        keyboard.hook(on_key_event)
        
        try:
            keyboard.wait(self.stop_key)
        except KeyboardInterrupt:
            print(f"\nLogging stopped by '{self.stop_key}'")
        finally:
            self.stop_logging()

    def _log_key_down(self, key: str):
        now = time.time()
        self.currently_pressed.add(key)
        self.key_press_times[key] = now
        self._log_action(key, 'press', now)

    def _log_key_up(self, key: str):
        now = time.time()
        if key in self.currently_pressed:
            self.currently_pressed.remove(key)
            hold_duration = now - self.key_press_times.get(key, now)
            self._log_action(key, 'release', now, hold_duration)

    def _log_action(self, action: str, event_type: str, timestamp: float, hold_duration: float = 0):
        time_delta = timestamp - self.last_action_time if self.last_action_time else 0
        
        action_data = {
            'timestamp': timestamp,
            'datetime': str(datetime.fromtimestamp(timestamp)),
            'action': action,
            'event_type': event_type,
            'hold_duration': hold_duration,
            'time_since_last': time_delta,
            'combo_size': len(self.currently_pressed)
        }

        self.data.append(action_data)
        
        if event_type == 'press':
            self.last_action_time = timestamp

        print(f"[{len(self.data)}] {action.upper()}-{event_type} "
              f"(DeltaTime: {time_delta:.2f}s, Hold: {hold_duration:.2f}s)")

    def _get_summary(self):
        if not self.data:
            return {}
        
        press_events = [d for d in self.data if d['event_type'] == 'press']
        hold_events = [d for d in self.data if d['hold_duration'] > 0]
        
        return {
            'total_events': len(self.data),
            'total_presses': len(press_events),
            'session_duration': self.data[-1]['timestamp'] - self.data[0]['timestamp'],
            'avg_hold_duration': np.mean([d['hold_duration'] for d in hold_events]) if hold_events else 0,
            'key_counts': {k: sum(1 for d in press_events if d['action'] == k) for k in self.valid_keys}
        }

    def _convert_to_training_format(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.data:
            return np.array([]), np.array([])

        X, Y = [], []
        press_events = [d for d in self.data if d['event_type'] == 'press']
        
        action_to_onehot = {k: [1 if i == idx else 0 for i in range(len(self.valid_keys))] 
                           for idx, k in enumerate(self.valid_keys)}

        for i in range(1, len(press_events)):
            current = press_events[i]
            previous = press_events[i-1]
            
            if current['action'] == previous['action']:
                continue

            features = [
                min(current['time_since_last'], 2.0),
                current['combo_size'] / len(self.valid_keys)
            ]
            
            features.extend(action_to_onehot[previous['action']])
            
            X.append(features)
            Y.append(action_to_onehot[current['action']])

        return np.array(X), np.array(Y)

    @classmethod
    def load_training_data(cls, path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(path, "r") as f:
            data = json.load(f)
        
        X = np.array(data['training_data']['X'])
        Y = np.array(data['training_data']['Y'])
        
        return X, Y