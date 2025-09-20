import numpy as np
import time
import random
import keyboard

from .class_MathLib import MathLib

class KeyboardPredictor:
    def __init__(self, network, key_order: list[str]):
        self.net = network
        self.key_order = key_order

    def predict_next_key(self, prev_action: str, time_norm: float) -> str:
        velocity = 1.0 / time_norm if time_norm > 0 else 0.0
        max_velocity = 1.0 / 0.1
        vel_norm = min(velocity / max_velocity, 1.0) * 2 - 1

        prev_one_hot = [1 if prev_action.lower() == k else 0 for k in self.key_order]
        X_test = np.array([[time_norm, vel_norm, *prev_one_hot]])

        pred_idx = self.net.predict(X_test)[0]  
        return self.key_order[pred_idx]

    def auto_press_keys(predictor, stop_key='q'):
        stop_key = stop_key.lower()
        prev_key = predictor.key_order[0]
        is_running = False
        key_pressed = False
        
        print(f"Auto-press ready. Press '{stop_key.upper()}' to start/stop")

        try:
            while True:
                if keyboard.is_pressed(stop_key):
                    if not key_pressed:
                        key_pressed = True
                        is_running = not is_running
                        
                        if is_running:
                            print("Auto-press started")
                            keyboard.press(prev_key)
                        else:
                            print("Auto-press paused")
                            keyboard.release(prev_key)
                else:
                    key_pressed = False

                if is_running:
                    delta = 0.1
                    normalized_delta = min(delta / 10.0, 1.0)

                    next_key = predictor.predict_next_key(prev_key, normalized_delta)

                    if next_key != prev_key:
                        keyboard.release(prev_key)
                        keyboard.press(next_key)
                        prev_key = next_key

                time.sleep(random.uniform(0.05, 0.3))

        except KeyboardInterrupt:
            print("\nStopping auto-press (Ctrl+C)")
        finally:
            if is_running:
                keyboard.release(prev_key)
            print("Auto-press ended")