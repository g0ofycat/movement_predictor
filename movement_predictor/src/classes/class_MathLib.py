import numpy as np

class MathLib:
    # ====== ACTIVATION FUNCTIONS ======
    
    @classmethod
    def ReLU(cls, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @classmethod
    def GeLU(cls, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
      
    @classmethod
    def Sigmoid(cls, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    # ====== ACTIVATION FUNCTION (DERIVATIVES) ======
    
    @classmethod
    def ReLU_derivative(cls, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @classmethod
    def Sigmoid_derivative(cls, x: np.ndarray) -> np.ndarray:
        s = cls.Sigmoid(x)
        return s * (1 - s)
    
    @classmethod
    def GeLU_derivative(cls, x: np.ndarray) -> np.ndarray:
        tanh_term = np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))
        return 0.5 * (1 + tanh_term + x*(1 - tanh_term**2) * np.sqrt(2/np.pi)*(1 + 3*0.044715*x**2))
      
    # ====== TRAINING & OUTPUT ======
    
    @classmethod
    def CrossEntropyLoss(cls, actual: np.ndarray, predicted: np.ndarray) -> float:
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return -np.sum(actual * np.log(predicted + epsilon))
    
    @classmethod
    def Softmax(cls, inputs: np.ndarray) -> np.ndarray:
        inputs = np.array(inputs, ndmin=1)
        max_value = np.max(inputs)
        exp_values = np.exp(inputs - max_value)
        return exp_values / np.sum(exp_values)
          
    @classmethod
    def MeanSquaredError(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean((y_true - y_pred)**2)