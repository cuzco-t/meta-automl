"""
Neural network model loader and inference module.
"""
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]):
        """
        Initialize the DQN network.
        
        Args:
            input_size: Input dimension (280)
            output_size: Output dimension (95)
            hidden_layers: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ModelLoader:
    """Handles loading and running inference on the trained DQN model."""
    
    def __init__(self, 
                 model_path: str,
                 model_info_path: str,
                 device: str = 'cpu'):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the .pt model file
            model_info_path: Path to the model_info.json file
            device: Device to load the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.model_info = self._load_json(model_info_path)
        self.model = self._build_and_load_model(model_path)
        self.model.eval()
    
    @staticmethod
    def _load_json(path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_and_load_model(self, model_path: str) -> DQNNetwork:
        """Build and load the model."""
        input_size = self.model_info['input_size']
        output_size = self.model_info['output_size']
        hidden_layers = self.model_info['hidden_layers']
        
        model = DQNNetwork(input_size, output_size, hidden_layers)
        model.to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        
        return model
    
    def predict(self, input_vector: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run inference and return probabilities and argmax.
        
        Args:
            input_vector: Input vector of shape (280,)
        
        Returns:
            Tuple of (probabilities, argmax_neuron)
        """
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to numpy
        output_np = output.cpu().numpy()[0]
        
        # Apply softmax to get probabilities
        exp_output = np.exp(output_np - np.max(output_np))  # For numerical stability
        probabilities = exp_output / exp_output.sum()
        
        argmax_neuron = np.argmax(probabilities)
        
        return probabilities, int(argmax_neuron)
