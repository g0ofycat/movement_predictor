import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# === IMPORTS ===

from src.classes.class_FFN import FeedForwardNetwork
from src.classes.class_KeyboardPredictor import KeyboardPredictor
from src.classes.class_MovementLogger import MovementLogger
from src.config.settings import settings

# === NETWORK ===

X, Y = MovementLogger.load_training_data(settings['settings']['movement_data_path'])

input_dim = X.shape[1]
output_dim = Y.shape[1]
hidden_dim = 64
hidden_layers = 3
dropout_rate = 0.0
epochs = 10000
learning_rate = 0.001

network = FeedForwardNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_layers=hidden_layers, dropout_rate=dropout_rate)

# === TRAINING ===

"""
network.train(X, Y, epochs=epochs, learning_rate=learning_rate)

network.save_model(settings['settings']['model_data_path'])
"""

# === PREDICT ===

network.load_model(settings['settings']['model_data_path'])

predictor = KeyboardPredictor(network, settings['settings']['valid_keys']) 

# print(network.get_model_info())

predictor.auto_press_keys()