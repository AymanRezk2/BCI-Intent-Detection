"""
Unified EEG Processing Pipeline

A complete end-to-end pipeline for processing EEG signals and running
both EEGNet (startle blink detection) and MIRepNet (motor imagery classification) models.
"""

from .pipeline import UnifiedEEGPipeline, create_pipeline
from .models_integration import ModelLoader, EEGNET_CLASS_LABELS, MIREPNET_CLASS_LABELS
from .preprocessing import (
    preprocess_for_eegnet,
    preprocess_for_mirepnet,
    apply_bandpass_filter,
    normalize_data,
    resample_data
)
from .utils import (
    detect_eeg_channels,
    extract_fp1_fp2,
    load_eeg_data
)

__version__ = "1.0.0"
__all__ = [
    'UnifiedEEGPipeline',
    'create_pipeline',
    'ModelLoader',
    'EEGNET_CLASS_LABELS',
    'MIREPNET_CLASS_LABELS',
    'preprocess_for_eegnet',
    'preprocess_for_mirepnet',
    'apply_bandpass_filter',
    'normalize_data',
    'resample_data',
    'detect_eeg_channels',
    'extract_fp1_fp2',
    'load_eeg_data'
]

