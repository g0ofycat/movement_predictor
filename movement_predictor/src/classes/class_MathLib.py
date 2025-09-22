import numpy as np

class MathLib:
    # ====== ACTIVATION FUNCTIONS ======
    
    @classmethod
    def ReLU(cls, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @classmethod
    def LeakyReLU(cls, x: np.ndarray, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @classmethod
    def GeLU(cls, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
      
    @classmethod
    def Sigmoid(cls, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    # ====== ACTIVATION FUNCTION (DERIVATIVES) ======
    
    @classmethod
    def ReLU_derivative(cls, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @classmethod
    def LeakyReLU_derivative(cls, x: np.ndarray, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @classmethod
    def GeLU_derivative(cls, x: np.ndarray) -> np.ndarray:
        tanh_term = np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))
        return 0.5 * (1 + tanh_term + x*(1 - tanh_term**2) * np.sqrt(2/np.pi)*(1 + 3*0.044715*x**2))
    
    @classmethod
    def Sigmoid_derivative(cls, x: np.ndarray) -> np.ndarray:
        s = cls.Sigmoid(x)
        return s * (1 - s)
      
    # ====== OUTPUT & LOSS FUNCTIONS ======
    
    @classmethod
    def Softmax(cls, inputs: np.ndarray) -> np.ndarray:
        max_values = np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs - max_values)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @classmethod
    def MeanSquaredError(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)
    
    @classmethod
    def CrossEntropyLoss(cls, actual: np.ndarray, predicted: np.ndarray) -> float:
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        n_samples = predicted.shape[0]
        loss = -np.sum(actual * np.log(predicted)) / n_samples
        return loss

    @classmethod
    def BinaryCrossEntropyLoss(cls, actual: np.ndarray, predicted: np.ndarray) -> float:
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        n_samples = actual.shape[0]
        loss = -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)) / n_samples
        return loss
    
    # ====== LOSS FUNCTION (DERIVATIVES) ======

    @classmethod
    def CrossEntropyLoss_derivative(cls, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        n_samples = actual.shape[0]
        return (predicted - actual) / n_samples

    @classmethod
    def BinaryCrossEntropyLoss_derivative(cls, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        n_samples = actual.shape[0]
        return ((1 - actual) / (1 - predicted) - actual / predicted) / n_samples