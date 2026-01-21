#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Student ID: 700762696
Student Name: Sai Madhuhas, Kotini

Instructions:
- Use only NumPy for computations
- Follow the docstring specifications carefully
- Run test_solution.py to verify your implementation
"""

import numpy as np
import struct
import gzip
from typing import List, Tuple, Dict
import pickle


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mnist(path='data/'):
    """
    Load MNIST dataset from files or download if not present.
    
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    import os
    import urllib.request
    
    # Create data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # MNIST file information
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files if not present
    base_url = 'https://github.com/fgnt/mnist/raw/master/'
    for file in files.values():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
    
    # Load data
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # Normalize to [0, 1]
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    X_train = load_images(os.path.join(path, files['train_images']))
    y_train = load_labels(os.path.join(path, files['train_labels']))
    X_test = load_images(os.path.join(path, files['test_images']))
    y_test = load_labels(os.path.join(path, files['test_labels']))
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# ============================================================================
# Layer Implementations
# ============================================================================

class Layer:
    """Base class for all layers."""
    def forward(self, X):
        """Forward pass - to be implemented by subclasses."""
        pass

    def backward(self, dL_dY):
        """Backward pass - to be implemented by subclasses."""
        pass

    def get_params(self):
        """Get layer parameters."""
        return {}

    def get_grads(self):
        """Get layer gradients."""
        return {}

    def set_params(self, params):
        """Set layer parameters."""
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        weight_init: Weight initialization method ('xavier', 'he', 'normal')
    """
    def __init__(self, input_dim, output_dim, weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights and biases
        if weight_init == 'xavier':
            scale = np.sqrt(2 / (input_dim + output_dim))
        elif weight_init == 'he':
            scale = np.sqrt(2 / input_dim)
        elif weight_init == 'normal':
            scale = 0.01
        else:
            raise ValueError("Invalid weight_init method")

        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
        
        # Storage for backward pass
        self.X = None
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward pass: Y = XW + b

        Args:
            X: Input data, shape (batch_size, input_dim)

        Returns:
            Y: Output data, shape (batch_size, output_dim)
        """
        # Store X for backward pass
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dL_dY):
        """
        Backward pass: compute gradients.

        Args:
            dL_dY: Gradient of loss w.r.t. output, shape (batch_size, output_dim)

        Returns:
            dL_dX: Gradient of loss w.r.t. input, shape (batch_size, input_dim)
        """
        # Compute gradients
        self.dW = self.X.T @ dL_dY
        self.db = np.sum(dL_dY, axis=0)
        return dL_dY @ self.W.T
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']

    def get_grads(self):
        return {'W': self.dW, 'b': self.db}


# ============================================================================
# Activation Functions
# ============================================================================

class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self):
        self.cache = None


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)

        Args:
            X: Input data

        Returns:
            Output after applying ReLU
        """
        # Store input for backward pass
        self.cache = X
        return np.maximum(0, X)
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = 1 if x > 0 else 0

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        return dL_dY * (self.cache > 0)


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = 1 / (1 + exp(-x))

        Args:
            X: Input data

        Returns:
            Output after applying sigmoid
        """
        # Store input for backward pass
        self.cache = X
        return 1 / (1 + np.exp(-X))
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = f(x) * (1 - f(x))

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        sigmoid_output = 1 / (1 + np.exp(-self.cache))
        return dL_dY * sigmoid_output * (1 - sigmoid_output)


class Softmax(Activation):
    """Softmax activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x_i) = exp(x_i) / sum(exp(x))

        Args:
            X: Input data, shape (batch_size, num_classes)

        Returns:
            Output probabilities, shape (batch_size, num_classes)
        """
        # Subtract max for numerical stability
        shifted_X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(shifted_X)
        self.cache = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.cache
    
    def backward(self, dL_dY):
        """
        Backward pass for softmax.

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        # For softmax, the Jacobian matrix is complex.
        # However, when combined with cross-entropy loss,
        # the gradient simplifies to (y_pred - y_true).
        # Here, we assume dL_dY is already (y_pred - y_true).
        return dL_dY


# ============================================================================
# Loss Functions
# ============================================================================

class Loss:
    """Base class for loss functions."""
    def compute(self, y_pred, y_true):
        """Compute loss - to be implemented by subclasses."""
        pass

    def gradient(self, y_pred, y_true):
        """Compute loss gradient - to be implemented by subclasses."""
        pass


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def compute(self, y_pred, y_true):
        """
        Compute MSE loss: L = 0.5 * mean((y_pred - y_true)^2)

        Args:
            y_pred: Predictions, shape (batch_size, num_features)
            y_true: True values, shape (batch_size, num_features)

        Returns:
            Scalar loss value
        """
        return 0.5 * np.mean((y_pred - y_true)**2)
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.

        Args:
            y_pred: Predictions
            y_true: True values

        Returns:
            Gradient w.r.t. predictions
        """
        return (y_pred - y_true) / y_pred.shape[0]


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification."""
    
    def compute(self, y_pred, y_true):
        """
        Compute cross-entropy loss: L = -mean(sum(y_true * log(y_pred)))

        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of cross-entropy loss.

        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot)

        Returns:
            Gradient w.r.t. predictions
        """
        return (y_pred - y_true) / y_pred.shape[0]


# ============================================================================
# Optimizers
# ============================================================================

class Optimizer:
    """Base class for optimizers."""
    def update(self, params, grads):
        """Update parameters - to be implemented by subclasses."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using vanilla SGD.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for key in params:
            params[key] = params[key].astype(float) - self.lr * grads[key]


class Momentum(Optimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key] = params[key].astype(float) + self.velocity[key]


# ============================================================================
# Neural Network Class
# ============================================================================

class NeuralNetwork:
    """
    Modular neural network implementation.
    
    Example usage:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.01))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """Configure the model for training."""
        self.loss_fn = loss
        self.optimizer = optimizer
    
    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input data

        Returns:
            Output of the network
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dL_dY):
        """
        Backward propagation through all layers.

        Args:
            dL_dY: Gradient of loss w.r.t. network output
        """
        for layer in reversed(self.layers):
            dL_dY = layer.backward(dL_dY)
    
    def update_params(self):
        """Update parameters of all trainable layers using the optimizer."""
        for layer in self.layers:
            if hasattr(layer, 'get_params') and hasattr(layer, 'get_grads') and hasattr(layer, 'set_params'):
                params = layer.get_params()
                grads = layer.get_grads()
                self.optimizer.update(params, grads)
                layer.set_params(params)
    
    def fit(self, X_train, y_train, epochs, batch_size, 
            X_val=None, y_val=None, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'train_acc': [], 
                   'val_loss': [], 'val_acc': []}
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0
            epoch_acc = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.loss_fn.compute(y_pred, y_batch)
                epoch_loss += loss

                # Compute accuracy
                pred_labels = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                acc = np.mean(pred_labels == true_labels)
                epoch_acc += acc

                # Backward pass
                dL_dy = self.loss_fn.gradient(y_pred, y_batch)
                self.backward(dL_dy)

                # Update parameters
                self.update_params()

            # Average over batches
            epoch_loss /= n_batches
            epoch_acc /= n_batches

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
                if X_val is not None:
                    print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
        return history
    
    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X: Input data

        Returns:
            Predictions (class indices for classification)
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Args:
            X: Input data
            y: True labels

        Returns:
            loss, accuracy
        """
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y_pred, y)
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        acc = np.mean(pred_labels == true_labels)
        return loss, acc

    def load_weights_from_params(self, params):
        """Load weights from params dictionary."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                layer_params = {}
                for key in layer.get_params():
                    layer_params[key] = params[f'layer_{i}_{key}']
                layer.set_params(layer_params)

    def save_weights(self, filename):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                weights[f'layer_{i}'] = layer.get_params()
        np.savez(filename, **weights)
    
    def load_weights(self, filename):
        """Load model weights from file."""
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in weights:
                layer.set_params(weights[f'layer_{i}'])


# ============================================================================
# Gradient Checking
# ============================================================================

def gradient_check(model, X, y, epsilon=1e-7):
    """
    Verify gradients using finite differences.

    Args:
        model: Neural network model
        X: Sample input data
        y: Sample labels
        epsilon: Small value for numerical differentiation

    Returns:
        Dictionary with gradient checking results
    """
    # Compute analytical gradients
    model.forward(X)
    y_pred = model.forward(X)
    loss = model.loss_fn.compute(y_pred, y)
    dL_dy = model.loss_fn.gradient(y_pred, y)
    model.backward(dL_dy)

    # Collect parameters and gradients
    params = {}
    grads = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_params') and hasattr(layer, 'get_grads'):
            layer_params = layer.get_params()
            layer_grads = layer.get_grads()
            for key in layer_params:
                params[f'layer_{i}_{key}'] = layer_params[key]
                grads[f'layer_{i}_{key}'] = layer_grads[key]

    # Numerical gradient checking
    relative_errors = {}
    for key in params:
        param = params[key]
        grad = grads[key]
        numerical_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param[idx]

            param[idx] = original_value + epsilon
            plus_loss = model.loss_fn.compute(model.forward(X), y)

            param[idx] = original_value - epsilon
            minus_loss = model.loss_fn.compute(model.forward(X), y)

            param[idx] = original_value

            numerical_grad[idx] = (plus_loss - minus_loss) / (2 * epsilon)
            it.iternext()

        # Compute relative error
        numerator = np.linalg.norm(grad - numerical_grad)
        denominator = np.linalg.norm(grad) + np.linalg.norm(numerical_grad)
        relative_error = numerator / denominator if denominator != 0 else 0
        relative_errors[key] = relative_error

    return relative_errors


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Create model
    print("Building neural network...")
    model = NeuralNetwork()
    
    # Build the network architecture
    # Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax
    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())
    
    # Compile model with CrossEntropyLoss and SGD optimizer
    model.compile(loss=CrossEntropyLoss(), optimizer=SGD(learning_rate=0.01))
    
    # Train the model
    print("Training model...")
    history = model.fit(X_train, y_train_oh, epochs=10, batch_size=32, X_val=X_test, y_val=y_test_oh, verbose=True)
    
    # Evaluate on test set
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Save model weights
    model.save_weights('model_weights.npz')
    
    # Save training log
    with open('training_log.txt', 'w') as f:
        f.write("Training Log\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}: Loss={history['train_loss'][epoch]:.4f}, Acc={history['train_acc'][epoch]:.4f}\n")
        f.write(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
    
    # Save sample predictions
    predictions = model.predict(X_test[:100])
    np.savetxt('predictions_sample.txt', predictions, fmt='%d')
    
    print("Training complete!")