"""
Utility functions for EEG data processing pipeline.
Includes channel detection, data loading, and helper functions.
"""

import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import warnings


def detect_eeg_channels(data: np.ndarray, channel_names: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Auto-detect EEG channels from data or channel names.
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data array. Shape can be (n_samples, n_channels) or (n_channels, n_samples)
    channel_names : Optional[List[str]]
        List of channel names if available
        
    Returns:
    --------
    Dict[str, int] : Dictionary mapping channel names to indices
        Keys: 'FP1', 'FP2', and any other detected channels
    """
    channel_map = {}
    
    # If channel names are provided, use them
    if channel_names is not None:
        channel_names_upper = [name.upper().strip() for name in channel_names]
        
        # Look for FP1 and FP2 (various naming conventions)
        fp1_variants = ['FP1', 'Fp1', 'FP1.', 'Fp1.', 'Fp1-Ref', 'FP1-Ref']
        fp2_variants = ['FP2', 'Fp2', 'FP2.', 'Fp2.', 'Fp2-Ref', 'FP2-Ref']
        
        for idx, ch_name in enumerate(channel_names_upper):
            # Check for FP1
            if any(variant in ch_name for variant in fp1_variants):
                channel_map['FP1'] = idx
            # Check for FP2
            elif any(variant in ch_name for variant in fp2_variants):
                channel_map['FP2'] = idx
            # Store all channels
            channel_map[ch_name] = idx
    
    # If no channel names, infer from data shape
    # Assume data is (n_samples, n_channels) or (n_channels, n_samples)
    if len(data.shape) == 2:
        n_channels = min(data.shape)
        # If we have 2 channels and no names, assume they are FP1 and FP2
        if n_channels == 2 and 'FP1' not in channel_map:
            channel_map['FP1'] = 0
            channel_map['FP2'] = 1
    
    return channel_map


def extract_fp1_fp2(data: np.ndarray, channel_map: Dict[str, int]) -> np.ndarray:
    """
    Extract FP1 and FP2 channels from EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data. Shape: (n_samples, n_channels) or (n_channels, n_samples)
    channel_map : Dict[str, int]
        Dictionary mapping channel names to indices
        
    Returns:
    --------
    np.ndarray : Extracted FP1 and FP2 data. Shape: (n_samples, 2) or (2, n_samples)
        Maintains the original data orientation
    """
    if 'FP1' not in channel_map or 'FP2' not in channel_map:
        raise ValueError("FP1 and/or FP2 channels not found in channel_map")
    
    fp1_idx = channel_map['FP1']
    fp2_idx = channel_map['FP2']
    
    # Determine data orientation
    if len(data.shape) == 2:
        # Check if channels are in first or second dimension
        if data.shape[0] < data.shape[1]:  # Likely (n_channels, n_samples)
            fp_data = np.array([data[fp1_idx, :], data[fp2_idx, :]])
        else:  # Likely (n_samples, n_channels)
            fp_data = data[:, [fp1_idx, fp2_idx]]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    return fp_data


def load_eeg_data(file_path: Union[str, Path], 
                  file_type: Optional[str] = None) -> Tuple[np.ndarray, List[str], float]:
    """
    Load EEG data from various file formats.
    
    Parameters:
    -----------
    file_path : Union[str, Path]
        Path to the EEG data file
    file_type : Optional[str]
        File type hint ('csv', 'edf', 'fif', 'npy', 'mat'). If None, auto-detect from extension.
        
    Returns:
    --------
    Tuple[np.ndarray, List[str], float]
        - data: EEG data array
        - channel_names: List of channel names
        - sampling_rate: Sampling frequency in Hz
    """
    file_path = Path(file_path)
    
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    if file_type == 'csv':
        # CSV format: time, channel1, channel2, ...
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        # Assume first column is time
        channel_names = [f'CH{i}' for i in range(data.shape[1] - 1)]
        # Default sampling rate for CSV (can be overridden)
        sampling_rate = 250.0
        return data[:, 1:], channel_names, sampling_rate
    
    elif file_type in ['edf', 'fif']:
        # MNE format
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if file_type == 'edf':
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            else:  # fif
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        channel_names = raw.ch_names
        sampling_rate = raw.info['sfreq']
        return data, channel_names, sampling_rate
    
    elif file_type == 'npy':
        # NumPy array format
        data = np.load(file_path)
        # Assume default channel names
        if len(data.shape) == 2:
            n_channels = min(data.shape)
            channel_names = [f'CH{i}' for i in range(n_channels)]
        else:
            raise ValueError(f"Unsupported data shape for .npy: {data.shape}")
        # Default sampling rate
        sampling_rate = 250.0
        return data, channel_names, sampling_rate
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def validate_data_shape(data: np.ndarray, expected_shape: Tuple, data_name: str = "Data") -> None:
    """
    Validate that data has the expected shape.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to validate
    expected_shape : Tuple
        Expected shape (can include None for variable dimensions)
    data_name : str
        Name for error messages
    """
    if len(data.shape) != len(expected_shape):
        raise ValueError(
            f"{data_name} shape mismatch: got {data.shape}, expected {expected_shape}"
        )
    
    for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{data_name} dimension {i} mismatch: got {actual}, expected {expected}"
            )


def synchronize_data(data1: np.ndarray, data2: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize two data arrays by trimming to the shorter length along specified axis.
    
    Parameters:
    -----------
    data1 : np.ndarray
        First data array
    data2 : np.ndarray
        Second data array
    axis : int
        Axis along which to synchronize (default: 0)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Synchronized data arrays
    """
    min_length = min(data1.shape[axis], data2.shape[axis])
    
    # Create slice indices
    slices1 = [slice(None)] * len(data1.shape)
    slices2 = [slice(None)] * len(data2.shape)
    slices1[axis] = slice(0, min_length)
    slices2[axis] = slice(0, min_length)
    
    return data1[tuple(slices1)], data2[tuple(slices2)]

