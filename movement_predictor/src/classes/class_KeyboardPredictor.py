import numpy as np

from .class_MathLib import MathLib

class KeyboardPredictor:
    def __init__(self, network, key_order: list[str]):
        self.net = network
        self.key_order = key_order

    def predict_next_key(self, prev_action: str, time_since_last: float) -> str:
        prev_one_hot = [1 if prev_action.lower() == k else 0 for k in self.key_order]
        X_test = np.array([[time_since_last, *prev_one_hot]])
        logits = self.net.predict(X_test)
        probs = np.array([MathLib.Softmax(row) for row in logits])
        predicted_index = np.argmax(probs[0])

        return self.key_order[predicted_index]