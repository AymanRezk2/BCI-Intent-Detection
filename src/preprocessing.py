"""
Unified preprocessing functions for EEG signals.
Combines preprocessing steps from both models (EEGNet and MIRepNet).
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from typing import Tuple, Optional
import warnings


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Parameters:
    -----------
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order (default: 4)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data: np.ndarray, 
                          lowcut: float, 
                          highcut: float, 
                          fs: float, 
                          order: int = 4,
                          axis: int = -1) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data to filter
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order (default: 4)
    axis : int
        Axis along which to apply filter (default: -1, last axis)
        
    Returns:
    --------
    np.ndarray
        Filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    # Apply filter along specified axis
    if axis == -1:
        filtered_data = filtfilt(b, a, data, axis=axis)
    else:
        # Move axis to last position, filter, then move back
        data_swapped = np.moveaxis(data, axis, -1)
        filtered_data = filtfilt(b, a, data_swapped, axis=-1)
        filtered_data = np.moveaxis(filtered_data, -1, axis)
    
    return filtered_data


def resample_data(data: np.ndarray, 
                  original_fs: float, 
                  target_fs: float,
                  axis: int = -1) -> np.ndarray:
    """
    Resample data to target sampling frequency using linear interpolation.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to resample
    original_fs : float
        Original sampling frequency
    target_fs : float
        Target sampling frequency
    axis : int
        Axis along which to resample (default: -1)
        
    Returns:
    --------
    np.ndarray
        Resampled data
    """
    if original_fs == target_fs:
        return data
    
    # Calculate new length
    original_length = data.shape[axis]
    new_length = int(original_length * target_fs / original_fs)
    
    if new_length <= 0:
        raise ValueError(f"Invalid resampling: original_fs={original_fs}, target_fs={target_fs}")
    
    # Create indices for interpolation
    original_indices = np.arange(original_length, dtype=np.float32)
    new_indices = np.linspace(0, original_length - 1, new_length, dtype=np.float32)
    
    # Apply interpolation along specified axis
    if len(data.shape) == 1:
        # 1D case
        resampled = np.interp(new_indices, original_indices, data)
    elif axis == -1 or axis == len(data.shape) - 1:
        # Resample along last axis
        resampled = np.zeros((*data.shape[:-1], new_length), dtype=data.dtype)
        # Use scipy's resample for better quality, or fall back to interp
        try:
            from scipy.signal import resample
            # Resample each slice along the last axis
            for idx in np.ndindex(data.shape[:-1]):
                resampled[idx] = resample(data[idx], new_length, axis=0)
        except ImportError:
            # Fall back to linear interpolation
            for idx in np.ndindex(data.shape[:-1]):
                resampled[idx] = np.interp(new_indices, original_indices, data[idx])
    else:
        # Move axis to last position, resample, then move back
        data_swapped = np.moveaxis(data, axis, -1)
        resampled_shape = (*data_swapped.shape[:-1], new_length)
        resampled = np.zeros(resampled_shape, dtype=data.dtype)
        
        # Resample each slice
        try:
            from scipy.signal import resample
            for idx in np.ndindex(data_swapped.shape[:-1]):
                resampled[idx] = resample(data_swapped[idx], new_length, axis=0)
        except ImportError:
            for idx in np.ndindex(data_swapped.shape[:-1]):
                resampled[idx] = np.interp(new_indices, original_indices, data_swapped[idx])
        
        resampled = np.moveaxis(resampled, -1, axis)
    
    return resampled


def apply_common_average_reference(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data. Shape: (n_channels, n_samples) or (n_samples, n_channels)
    axis : int
        Axis along which channels are located (0 for channels-first, -1 for channels-last)
        
    Returns:
    --------
    np.ndarray
        CAR-referenced data
    """
    if axis == 0:
        # Channels are in first dimension
        mean = np.mean(data, axis=0, keepdims=True)
        return data - mean
    elif axis == -1:
        # Channels are in last dimension
        mean = np.mean(data, axis=-1, keepdims=True)
        return data - mean
    else:
        raise ValueError(f"Unsupported axis for CAR: {axis}")


def normalize_data(data: np.ndarray, 
                   method: str = 'zscore',
                   axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to normalize
    method : str
        Normalization method: 'zscore', 'minmax', 'robust'
    axis : Optional[int]
        Axis along which to compute statistics. If None, normalize globally.
        
    Returns:
    --------
    np.ndarray
        Normalized data
    """
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(data, axis=axis, keepdims=True) if axis is not None else np.mean(data)
        std = np.std(data, axis=axis, keepdims=True) if axis is not None else np.std(data)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (data - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(data, axis=axis, keepdims=True) if axis is not None else np.min(data)
        max_val = np.max(data, axis=axis, keepdims=True) if axis is not None else np.max(data)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
        return (data - min_val) / range_val
    
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(data, axis=axis, keepdims=True) if axis is not None else np.median(data)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True) if axis is not None else np.percentile(data, 75)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True) if axis is not None else np.percentile(data, 25)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1, iqr)  # Avoid division by zero
        return (data - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_windows(data: np.ndarray, 
                   window_size: int, 
                   step_size: Optional[int] = None,
                   axis: int = 0) -> np.ndarray:
    """
    Create sliding windows from continuous EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        Continuous EEG data
    window_size : int
        Size of each window in samples
    step_size : Optional[int]
        Step size between windows. If None, use window_size (non-overlapping)
    axis : int
        Axis along which to create windows (default: 0)
        
    Returns:
    --------
    np.ndarray
        Windowed data. Shape: (n_windows, ..., window_size, ...)
    """
    if step_size is None:
        step_size = window_size
    
    # Get data length along specified axis
    data_length = data.shape[axis]
    
    # Calculate number of windows
    n_windows = (data_length - window_size) // step_size + 1
    
    if n_windows <= 0:
        raise ValueError(f"Window size {window_size} is larger than data length {data_length}")
    
    # Create indices for windows
    windows = []
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Create slice indices
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(start_idx, end_idx)
        windows.append(data[tuple(slices)])
    
    return np.array(windows)


def pad_data(data: np.ndarray, 
             target_length: int,
             axis: int = -1,
             mode: str = 'constant',
             constant_values: float = 0.0) -> np.ndarray:
    """
    Pad data to target length along specified axis.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to pad
    target_length : int
        Target length
    axis : int
        Axis along which to pad (default: -1)
    mode : str
        Padding mode: 'constant', 'edge', 'mean'
    constant_values : float
        Value for constant padding (default: 0.0)
        
    Returns:
    --------
    np.ndarray
        Padded data
    """
    current_length = data.shape[axis]
    
    if current_length >= target_length:
        # If data is longer, truncate instead of pad
        slices = [slice(None)] * len(data.shape)
        slices[axis] = slice(0, target_length)
        return data[tuple(slices)]
    
    pad_width = target_length - current_length
    
    # Create padding tuple
    pad_tuple = [(0, 0)] * len(data.shape)
    pad_tuple[axis] = (0, pad_width)
    
    if mode == 'constant':
        return np.pad(data, pad_tuple, mode='constant', constant_values=constant_values)
    elif mode == 'edge':
        return np.pad(data, pad_tuple, mode='edge')
    elif mode == 'mean':
        # Pad with mean value along the axis
        mean_val = np.mean(data, axis=axis, keepdims=True)
        return np.pad(data, pad_tuple, mode='constant', constant_values=mean_val)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def preprocess_for_eegnet(data: np.ndarray,
                          fs: float,
                          target_fs: float = 250.0,
                          window_size: int = 250,
                          normalize: bool = True) -> np.ndarray:
    """
    Preprocess data for EEGNet model (startle blink detection).
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data. Shape: (n_samples, 2) for FP1/FP2 or (2, n_samples)
    fs : float
        Current sampling frequency
    target_fs : float
        Target sampling frequency (default: 250.0 Hz)
    window_size : int
        Window size in samples (default: 250 = 1 second at 250 Hz)
    normalize : bool
        Whether to apply normalization (default: True)
        
    Returns:
    --------
    np.ndarray
        Preprocessed data. Shape: (n_windows, window_size, 2) or (n_windows, 2, window_size)
    """
    # Ensure data is (n_samples, 2)
    if data.shape[0] == 2 and data.shape[1] != 2:
        data = data.T
    
    # Resample if needed
    if fs != target_fs:
        data = resample_data(data, fs, target_fs, axis=0)
    
    # Normalize
    if normalize:
        data = normalize_data(data, method='zscore', axis=0)
    
    # Create windows
    windows = create_windows(data, window_size, step_size=window_size, axis=0)
    
    # Ensure shape is (n_windows, window_size, 2)
    if len(windows.shape) == 3 and windows.shape[1] == 2:
        windows = np.transpose(windows, (0, 2, 1))
    
    return windows.astype('float32')


def preprocess_for_mirepnet(data: np.ndarray,
                            fs: float,
                            target_fs: float = 128.0,
                            lowcut: float = 4.0,
                            highcut: float = 38.0,
                            target_length: int = 512,
                            apply_car: bool = True,
                            normalize: bool = True) -> np.ndarray:
    """
    Preprocess data for MIRepNet model (motor imagery classification).
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data. Shape: (n_channels, n_samples) or (n_samples, n_channels)
    fs : float
        Current sampling frequency
    target_fs : float
        Target sampling frequency (default: 128.0 Hz)
    lowcut : float
        Low cutoff frequency for bandpass filter (default: 4.0 Hz)
    highcut : float
        High cutoff frequency for bandpass filter (default: 38.0 Hz)
    target_length : int
        Target length in samples (default: 512 = 4 seconds at 128 Hz)
    apply_car : bool
        Whether to apply Common Average Reference (default: True)
    normalize : bool
        Whether to apply normalization (default: True)
        
    Returns:
    --------
    np.ndarray
        Preprocessed data. Shape: (n_channels, target_length)
    """
    # Ensure data is (n_channels, n_samples)
    if len(data.shape) == 2:
        if data.shape[0] > data.shape[1]:
            # Likely (n_samples, n_channels), transpose
            data = data.T
    
    # Apply bandpass filter
    data = apply_bandpass_filter(data, lowcut, highcut, fs, axis=1)
    
    # Resample if needed
    if fs != target_fs:
        data = resample_data(data, fs, target_fs, axis=1)
    
    # Apply CAR
    if apply_car:
        data = apply_common_average_reference(data, axis=0)
    
    # Pad or truncate to target length
    data = pad_data(data, target_length, axis=1, mode='constant')
    
    # Normalize
    if normalize:
        data = normalize_data(data, method='zscore', axis=1)
    
    return data.astype('float32')

