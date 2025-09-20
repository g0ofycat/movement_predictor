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
        
        current_key = random.choice(self.key_order)
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
                            keyboard.press(current_key)
                            last_press_time = current_time
                            self.last_action = current_key
                            self.last_action_time = current_time
                        else:
                            print("Auto-press paused")
                            for key in self.key_order:
                                keyboard.release(key)
                else:
                    key_pressed = False

                if is_running:
                    time_since_last = current_time - last_press_time

                    actual_interval = base_interval + random.uniform(-0.05, 0.05) if randomize_timing else base_interval
                    
                    if time_since_last >= actual_interval:
                        combo_size = 2 if random.random() < 0.1 else 1
                        
                        predicted_key = self.predict_next_key(current_key, time_since_last, combo_size)
                        
                        if predicted_key == current_key:
                            if random.random() < 0.3:
                                available_keys = [k for k in self.key_order if k != current_key]
                                predicted_key = random.choice(available_keys)
                                print(f"[FORCED CHANGE] From {current_key.upper()} to {predicted_key.upper()}")
                        
                        if predicted_key != current_key:
                            keyboard.release(current_key)
                            time.sleep(0.01)
                            keyboard.press(predicted_key)
                            
                            print(f"[{self.action_count}] {current_key.upper()} → {predicted_key.upper()}")
                            
                            current_key = predicted_key
                            last_press_time = current_time
                            self.last_action = current_key
                            self.last_action_time = current_time
                            self.action_count += 1
                        else:
                            last_press_time = current_time

                if randomize_timing:
                    sleep_time = random.uniform(0.01, 0.03)
                else:
                    sleep_time = 0.02
                    
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping auto-press")
        finally:
            for key in self.key_order:
                keyboard.release(key)
            print(f"Auto-press ended. Total actions: {self.action_count}")

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