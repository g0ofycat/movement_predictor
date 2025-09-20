import numpy as np
import time
import random
import keyboard
from typing import List

class KeyboardPredictor:
    def __init__(self, network, key_order: List[str]):
        self.net = network
        self.key_order = key_order
        self.last_action = None
        self.last_action_time = None
        self.action_count = 0

    def predict_next_key(self, prev_action: str, time_delta: float, combo_size: int = 1) -> str:
        features = [
            min(time_delta, 2.0),
            combo_size / len(self.key_order)
        ]

        prev_key_onehot = [1 if prev_action == k else 0 for k in self.key_order]
        features.extend(prev_key_onehot)
        
        X_test = np.array([features])
        pred_probs = self.net.predict(X_test)[0]

        pred_idx = np.argmax(pred_probs)
        return self.key_order[pred_idx]

    def reset_state(self):
        self.last_action = None
        self.last_action_time = None
        self.action_count = 0

    def auto_press_keys(self, stop_key='q', base_interval=0.15, randomize_timing=True):
        stop_key = stop_key.lower()
        prev_key = self.key_order[0]
        is_running = False
        key_pressed = False
        
        self.reset_state()
        last_press_time = time.time()
        
        print(f"Auto-press ready. Press '{stop_key.upper()}' to start / stop")

        try:
            while True:
                current_time = time.time()
                
                if keyboard.is_pressed(stop_key):
                    if not key_pressed:
                        key_pressed = True
                        is_running = not is_running
                        
                        if is_running:
                            print("Auto-press started")
                            keyboard.press(prev_key)
                            last_press_time = current_time
                            self.last_action = prev_key
                            self.last_action_time = current_time
                        else:
                            print("Auto-press paused")
                            keyboard.release(prev_key)
                else:
                    key_pressed = False

                if is_running:
                    time_since_last = current_time - last_press_time
                    
                    if time_since_last >= base_interval:
                        combo_size = 2 if random.random() < 0.1 else 1
                        
                        next_key = self.predict_next_key(prev_key, time_since_last, combo_size)
                        
                        if next_key != prev_key:
                            keyboard.release(prev_key)
                            keyboard.press(next_key)
                            
                            prev_key = next_key
                            last_press_time = current_time
                            self.action_count += 1
                            
                            print(f"[{self.action_count}] Switched to: {next_key.upper()}")
                
                if randomize_timing:
                    sleep_time = random.uniform(0.03, 0.07)
                else:
                    sleep_time = 0.05
                    
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping auto-press")
        finally:
            if is_running:
                keyboard.release(prev_key)
            print(f"Auto-press ended")

    def test_prediction_accuracy(self, test_data_x, test_data_y, num_samples=100):
        if len(test_data_x) == 0:
            print("No test data available")
            return 0.0
        
        sample_size = min(num_samples, len(test_data_x))
        indices = np.random.choice(len(test_data_x), sample_size, replace=False)
        
        correct = 0
        
        for i in indices:
            pred_probs = self.net.predict(np.array([test_data_x[i]]))[0]
            predicted_idx = np.argmax(pred_probs)
            actual_idx = np.argmax(test_data_y[i])
            
            if predicted_idx == actual_idx:
                correct += 1
        
        accuracy = correct / sample_size

        return accuracy

    def get_key_probabilities(self, prev_action: str, time_delta: float, combo_size: int = 1):
        features = [
            min(time_delta, 2.0),
            combo_size / len(self.key_order)
        ]
        
        prev_key_onehot = [1 if prev_action == k else 0 for k in self.key_order]
        features.extend(prev_key_onehot)
        
        X_test = np.array([features])
        probabilities = self.net.predict(X_test)[0]
        
        return {key: prob for key, prob in zip(self.key_order, probabilities)}