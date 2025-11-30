"""
Model integration module for loading and running both EEGNet and MIRepNet models.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. EEGNet model will not work.")


# ============================================================================
# MIRepNet Model Architecture (PyTorch)
# ============================================================================

class SpatioTemporalFeatureExtractor(nn.Module):
    """CNN-based Spatio-Temporal Feature Extractor for MIRepNet."""
    
    def __init__(self, in_channels: int = 64, time_steps: int = 512, embed_dim: int = 128):
        super(SpatioTemporalFeatureExtractor, self).__init__()
        
        # Spatial Convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32, stride=8, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(64, embed_dim, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x


class MIRepNet(nn.Module):
    """Full MIRepNet model combining CNN feature extraction and Transformer Encoder."""
    
    def __init__(self, 
                 in_channels: int = 64, 
                 time_steps: int = 512, 
                 embed_dim: int = 128, 
                 num_heads: int = 8, 
                 num_layers: int = 2, 
                 num_classes: int = 4):
        super(MIRepNet, self).__init__()
        
        self.feature_extractor = SpatioTemporalFeatureExtractor(
            in_channels=in_channels,
            time_steps=time_steps,
            embed_dim=embed_dim
        )
        
        # Determine sequence length dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, time_steps)
            dummy_output = self.feature_extractor(dummy_input)
            self.seq_len = dummy_output.shape[2]
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)  # (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.transformer_encoder(x)  # (B, L, C)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.global_pool(x)  # (B, C, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# Model Loaders
# ============================================================================

class ModelLoader:
    """Class to load and manage both models."""
    
    def __init__(self, 
                 eegnet_path: Optional[str] = None,
                 mirepnet_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize model loader.
        
        Parameters:
        -----------
        eegnet_path : Optional[str]
            Path to EEGNet Keras model file
        mirepnet_path : Optional[str]
            Path to MIRepNet PyTorch model file
        device : Optional[str]
            Device for PyTorch model ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.eegnet_model = None
        self.mirepnet_model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load EEGNet model
        if eegnet_path and TF_AVAILABLE:
            self.load_eegnet(eegnet_path)
        
        # Load MIRepNet model
        if mirepnet_path:
            self.load_mirepnet(mirepnet_path)
    
    def load_eegnet(self, model_path: str) -> None:
        """
        Load EEGNet Keras model.
        
        Parameters:
        -----------
        model_path : str
            Path to Keras model file (.keras or .h5)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot load EEGNet model.")
        
        try:
            self.eegnet_model = keras.models.load_model(model_path, compile=False)
            print(f"✅ EEGNet model loaded from {model_path}")
            print(f"   Input shape: {self.eegnet_model.input_shape}")
            print(f"   Output shape: {self.eegnet_model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load EEGNet model from {model_path}: {e}")
    
    def load_mirepnet(self, model_path: str, 
                      n_channels: int = 64,
                      n_timesteps: int = 512,
                      n_classes: int = 4) -> None:
        """
        Load MIRepNet PyTorch model.
        
        Parameters:
        -----------
        model_path : str
            Path to PyTorch model file (.pth)
        n_channels : int
            Number of input channels (default: 64)
        n_timesteps : int
            Number of timesteps (default: 512). 
            Note: Model can handle slight variations (e.g., 512-513) due to padding.
        n_classes : int
            Number of output classes (default: 4)
        """
        try:
            # Try to load with the specified timesteps
            # The model uses adaptive pooling, so it can handle slight variations
            self.mirepnet_model = MIRepNet(
                in_channels=n_channels,
                time_steps=n_timesteps,
                num_classes=n_classes
            )
            self.mirepnet_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.mirepnet_model.to(self.device)
            self.mirepnet_model.eval()
            print(f"✅ MIRepNet model loaded from {model_path}")
            print(f"   Input shape: (batch, {n_channels}, {n_timesteps})")
            print(f"   Output shape: (batch, {n_classes})")
            print(f"   Device: {self.device}")
            print(f"   Note: Model can handle timesteps in range [~480, ~550] due to padding")
        except Exception as e:
            raise RuntimeError(f"Failed to load MIRepNet model from {model_path}: {e}")


# ============================================================================
# Inference Functions
# ============================================================================

def predict_eegnet(model, data: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with EEGNet model.
    
    Parameters:
    -----------
    model : keras.Model
        Loaded EEGNet model
    data : np.ndarray
        Preprocessed data. Shape: (n_windows, 250, 2) or (1, 250, 2)
    threshold : float
        Classification threshold (default: 0.5)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - probabilities: Raw model output probabilities
        - predictions: Binary predictions (0 or 1)
    """
    if model is None:
        raise ValueError("EEGNet model not loaded")
    
    # Ensure data is float32
    data = data.astype('float32')
    
    # Add batch dimension if needed
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    
    # Run inference
    probabilities = model.predict(data, verbose=0)
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int).flatten()
    
    return probabilities.flatten(), predictions


def predict_mirepnet(model, data: np.ndarray, 
                     device: str = 'cpu',
                     return_probs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with MIRepNet model.
    
    Parameters:
    -----------
    model : nn.Module
        Loaded MIRepNet model
    data : np.ndarray
        Preprocessed data. Shape: (n_channels, n_timesteps) or (1, n_channels, n_timesteps)
    device : str
        Device to run inference on
    return_probs : bool
        Whether to return probabilities (default: True)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - probabilities: Class probabilities (if return_probs=True) or logits
        - predictions: Predicted class indices
    """
    if model is None:
        raise ValueError("MIRepNet model not loaded")
    
    # Ensure data is float32
    data = data.astype('float32')
    
    # Convert to torch tensor
    if len(data.shape) == 2:
        # Add batch dimension: (n_channels, n_timesteps) -> (1, n_channels, n_timesteps)
        data = torch.from_numpy(data).unsqueeze(0)
    else:
        data = torch.from_numpy(data)
    
    data = data.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(data)
        
        if return_probs:
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)
            probabilities = probs.cpu().numpy()
        else:
            probabilities = output.cpu().numpy()
        
        predictions = torch.argmax(output, dim=1).cpu().numpy()
    
    return probabilities, predictions


# ============================================================================
# Class Label Mappings
# ============================================================================

MIREPNET_CLASS_LABELS = {
    0: 'Forward',
    1: 'Left',
    2: 'Right',
    3: 'Stop'
}

EEGNET_CLASS_LABELS = {
    0: 'No Blink',
    1: 'Startle Blink'
}

