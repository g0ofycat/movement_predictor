import numpy as np
import json
from pathlib import Path

from .class_MathLib import MathLib

class FeedForwardNetwork:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int,
        dropout_rate: float = 0.0
    ) -> None:
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        self.weights.append(
            np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        )

        self.biases.append(np.zeros((1, hidden_dim)))

        for _ in range(hidden_layers - 1):
            self.weights.append(
                np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
            )

            self.biases.append(np.zeros((1, hidden_dim)))

        self.weights.append(
            np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        )

        self.biases.append(np.zeros((1, output_dim)))

    # ====== MAIN API ======

    def forward_propagation(
        self, x: np.ndarray, apply_dropout: bool = True
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        
        a = np.atleast_2d(x)
        
        activations: list[np.ndarray] = [a]
        pre_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray] = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_activations.append(z)

            if i < len(self.weights) - 1:
                a = MathLib.ReLU(z)
                
                if apply_dropout and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                    dropout_masks.append(dropout_mask)
                    a = a * dropout_mask
                else:
                    dropout_masks.append(np.ones_like(a))         
            else:
                a = z
                dropout_masks.append(np.ones_like(a))
                
            activations.append(a)

        return activations, pre_activations, dropout_masks

    def back_propagation(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        learning_rate: float = 0.01,
    ) -> None:
        
        x = np.atleast_2d(x)
        y_true = np.atleast_2d(y_true)
        m = x.shape[0]

        activations, pre_activations, dropout_masks = self.forward_propagation(x)
        
        delta = 2 * (activations[-1] - y_true) / m

        for layer in reversed(range(len(self.weights))):
            if layer < len(self.weights) - 1:
                delta = delta * MathLib.ReLU_derivative(pre_activations[layer])

                delta = delta * dropout_masks[layer]

            dW = activations[layer].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db

            if layer > 0:
                delta = delta @ self.weights[layer].T

    # ====== TRAINING ======

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        
        x_train = np.atleast_2d(x_train)
        y_train = np.atleast_2d(y_train)
        
        n_samples = x_train.shape[0]
        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                x_batch = x_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                self.back_propagation(x_batch, y_batch, learning_rate)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                y_pred = self.forward_propagation(x_train)[0][-1]
                loss = MathLib.MeanSquaredError(y_train, y_pred)
                loss_history.append(loss)
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")
        
        return loss_history

    def predict(self, x: np.ndarray) -> np.ndarray:
        activations, _, _ = self.forward_propagation(x, apply_dropout=False)

        return activations[-1]
    
    # ====== MODEL ======

    def save_model(self, filepath: str) -> None:
        filepath = Path(filepath)

        assert filepath.suffix == '.json', '[save_model]: Files must be .json'

        model_data = {
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate
            },
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        filepath = Path(filepath)

        assert filepath.suffix == '.json', '[load_model]: Files must be .json'
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        arch = model_data['architecture']
        self.input_dim = arch['input_dim']
        self.hidden_dim = arch['hidden_dim']
        self.output_dim = arch['output_dim']
        self.hidden_layers = arch['hidden_layers']
        self.dropout_rate = arch['dropout_rate']
        
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        
        print(f"Model loaded from {filepath}")

    def get_model_info(self) -> str:
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
        info = f"Neural Network Architecture:\n"
        info += f"Input dimensions: {self.input_dim}\n"
        info += f"Hidden layers: {self.hidden_layers} (Each with {self.hidden_dim} Neurons)\n"
        info += f"Output dimensions: {self.output_dim}\n"
        info += f"Dropout rate: {self.dropout_rate}\n"
        info += f"Total parameters: {total_params:,}\n"
        
        return info