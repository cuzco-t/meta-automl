"""
DQN Pipeline Executor: Encapsulates DQN model loading and action recommendation logic.

This class handles:
- Loading the trained DQN model
- Loading action mappings and phase information from CSV
- Querying the network for action recommendations given a vector and phase
- Returning ranked actions with activation probabilities and dimension mappings
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from neural_network import ModelLoader


class DQNPipeline:
    """
    Encapsulates DQN model and action recommendation logic.
    
    Provides a simple interface to query the network for recommended actions
    for a given phase, allowing the caller to handle success/failure logic externally.
    """
    
    # Vector configuration
    FIXED_DIMS = 185          # First 185 dimensions are fixed
    TOTAL_DIMS = 280          # Total input dimensions
    
    # Phase name mapping
    PHASE_NAMES = {
        0: "tratar_duplicados",
        1: "codificar_variables_binarias",
        2: "tratar_faltantes_numericos",
        3: "tratar_faltantes_strings",
        4: "codificar_variables_categoricas_rango_bajo",
        5: "codificar_variables_categoricas_rango_medio",
        6: "codificar_variables_categoricas_rango_alto",
        7: "tratar_outliers_numericos",
        8: "escalar_datos_numericos",
        9: "normalizar_datos_numericos",
        10: "crear_nueva_variable",
        11: "seleccionar_variables",
        12: "modelo_ml",
        13: "clasificacion",
        14: "regresion",
        15: "clustering",
    }
    
    def __init__(self,
                 model_path: str,
                 model_info_path: str,
                 action_mapping_path: str,
                 actions_csv_path: str,
                 device: str = 'cpu'):
        """
        Initialize the DQN Pipeline.
        
        Args:
            model_path: Path to the trained model (.pt file)
            model_info_path: Path to model_info.json
            action_mapping_path: Path to action_mapping.json (bidirectional mapping)
            actions_csv_path: Path to acciones.csv (phase and index information)
            device: Device to run model on ('cpu' or 'cuda')
        
        Raises:
            FileNotFoundError: If any required file is not found
            ValueError: If configuration is invalid
        """
        self.device = device
        
        # Load model
        try:
            self.model_loader = ModelLoader(model_path, model_info_path, device)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
        
        # Load action mapping
        try:
            with open(action_mapping_path, 'r') as f:
                action_map = json.load(f)
                self.idx_to_action = {int(k): v for k, v in action_map['idx_to_action'].items()}
                self.action_to_idx = action_map['action_to_idx']
        except Exception as e:
            raise ValueError(f"Failed to load action mapping: {e}")
        
        # Load actions CSV for phase mapping
        try:
            self.actions_df = pd.read_csv(actions_csv_path)
        except Exception as e:
            raise ValueError(f"Failed to load actions CSV: {e}")
        
        # Build phase-to-neurons mapping
        self.phase_to_neurons = self._build_phase_to_neurons()
        self.neuron_to_phase = self._build_neuron_to_phase()
        
        # Validate configuration
        self._validate_configuration()
    
    def _build_phase_to_neurons(self) -> Dict[int, List[int]]:
        """Build mapping from phase number to list of neuron indices."""
        mapping = {}
        for _, row in self.actions_df.iterrows():
            phase = row['fase']
            neuron_idx = row['index']
            
            if phase not in mapping:
                mapping[phase] = []
            mapping[phase].append(int(neuron_idx))
        
        # Sort neurons for each phase
        for phase in mapping:
            mapping[phase].sort()
        
        return mapping
    
    def _build_neuron_to_phase(self) -> Dict[int, int]:
        """Build mapping from neuron index to phase."""
        mapping = {}
        for _, row in self.actions_df.iterrows():
            mapping[int(row['index'])] = row['fase']
        return mapping
    
    def _validate_configuration(self):
        """Validate that configuration is correct."""
        # Check that all neurons in mapping are valid
        for phase, neurons in self.phase_to_neurons.items():
            for neuron in neurons:
                if neuron not in self.idx_to_action:
                    raise ValueError(f"Neuron {neuron} in phase {phase} not found in action mapping")
    
    def _calculate_activation_dimension(self, neuron_idx: int) -> int:
        """
        Calculate the dimension in the input vector to activate for a given neuron.
        
        Current strategy: dimension = FIXED_DIMS + neuron_idx
        This can be overridden if a different mapping is needed.
        
        Args:
            neuron_idx: Index of the neuron
        
        Returns:
            Dimension index to activate (0-based)
        """
        dimension = self.FIXED_DIMS + neuron_idx
        
        # Validate dimension is within bounds
        if dimension >= self.TOTAL_DIMS:
            raise ValueError(
                f"Calculated dimension {dimension} exceeds total dimensions {self.TOTAL_DIMS} "
                f"for neuron {neuron_idx}"
            )
        
        return dimension
    
    def get_actions_for_phase(self, 
                             input_vector: np.ndarray, 
                             phase: int) -> List[Dict]:
        """
        Get recommended actions for a given phase, ranked by activation probability.
        
        This is the main interface method. It:
        1. Validates input vector
        2. Passes vector through the network
        3. Gets activations for all neurons in the phase
        4. Returns ranked list with action info and dimension mappings
        
        Args:
            input_vector: Input vector (must be 280 dimensions)
            phase: Phase number (0-15)
        
        Returns:
            List of dictionaries, each containing:
            {
                'neuron_idx': int,              # Index of output neuron
                'action_name': str,             # Name of the action
                'activation': float,            # Raw activation value (0-1)
                'activation_percent': float,    # Activation percentage (0-100)
                'dimension_to_activate': int,   # Dimension to activate if action succeeds
                'phase': int                    # Phase number (for reference)
            }
            
            Sorted by activation_percent in descending order (highest first)
        
        Raises:
            ValueError: If phase is invalid or input vector has wrong shape
        """
        # Validate input
        if not isinstance(input_vector, np.ndarray):
            raise ValueError(f"Input vector must be numpy array, got {type(input_vector)}")
        
        if input_vector.shape[0] != self.TOTAL_DIMS:
            raise ValueError(
                f"Input vector must have {self.TOTAL_DIMS} dimensions, "
                f"got {input_vector.shape[0]}"
            )
        
        if phase not in self.phase_to_neurons:
            raise ValueError(
                f"Invalid phase {phase}. Valid phases: {sorted(self.phase_to_neurons.keys())}"
            )
        
        # Get predictions from model
        try:
            probabilities, _ = self.model_loader.predict(input_vector)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Get all valid neurons for this phase
        valid_neurons = self.phase_to_neurons[phase]
        
        # Create ranked list of actions
        ranked_actions = []
        
        for neuron_idx in valid_neurons:
            if neuron_idx >= len(probabilities):
                raise RuntimeError(
                    f"Neuron index {neuron_idx} exceeds model output size {len(probabilities)}"
                )
            
            # Get activation
            activation = float(probabilities[neuron_idx])
            
            # Get action name
            action_name = self.idx_to_action.get(neuron_idx)
            if action_name is None:
                raise RuntimeError(f"Action name not found for neuron {neuron_idx}")
            
            # Calculate dimension to activate
            try:
                dimension = self._calculate_activation_dimension(neuron_idx)
            except ValueError as e:
                raise RuntimeError(f"Failed to calculate dimension for neuron {neuron_idx}: {e}")
            
            # Add to list
            ranked_actions.append({
                'neuron_idx': neuron_idx,
                'action_name': action_name,
                'activation': activation,
                'activation_percent': activation * 100.0,
                'dimension_to_activate': dimension,
                'phase': phase
            })
        
        # Sort by activation (descending)
        ranked_actions.sort(key=lambda x: x['activation'], reverse=True)
        
        return ranked_actions
    
    def apply_action_to_vector(self, 
                              input_vector: np.ndarray, 
                              dimension: int,
                              value: float = 1.0) -> np.ndarray:
        """
        Apply an action to the input vector by activating a dimension.
        
        This is a convenience method that activates a dimension in the vector.
        It's provided for utility but can also be done externally.
        
        Args:
            input_vector: Input vector to modify (will be copied)
            dimension: Dimension to activate
            value: Value to set (default 1.0)
        
        Returns:
            Modified copy of the input vector
        
        Raises:
            ValueError: If dimension is out of bounds
        """
        if dimension < 0 or dimension >= self.TOTAL_DIMS:
            raise ValueError(
                f"Dimension {dimension} out of bounds [0, {self.TOTAL_DIMS-1}]"
            )
        
        updated_vector = input_vector.copy()
        updated_vector[dimension] = value
        return updated_vector
    
    def get_phase_name(self, phase: int) -> str:
        """Get human-readable name for a phase."""
        return self.PHASE_NAMES.get(phase, f"unknown_phase_{phase}")
    
    def get_available_phases(self) -> List[int]:
        """Get list of all available phases."""
        return sorted(self.phase_to_neurons.keys())
    
    def get_phase_action_count(self, phase: int) -> int:
        """Get number of actions available in a phase."""
        return len(self.phase_to_neurons.get(phase, []))


__all__ = ['DQNPipeline']
