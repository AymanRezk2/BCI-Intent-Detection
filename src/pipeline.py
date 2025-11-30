"""
Unified end-to-end pipeline for EEG signal processing and model inference.
Combines preprocessing and inference for both EEGNet and MIRepNet models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import warnings

try:
    from .utils import (
        detect_eeg_channels,
        extract_fp1_fp2,
        load_eeg_data,
        synchronize_data,
        validate_data_shape
    )
    from .preprocessing import (
        preprocess_for_eegnet,
        preprocess_for_mirepnet,
        resample_data,
        apply_bandpass_filter,
        apply_common_average_reference,
        normalize_data
    )
    from .models_integration import (
        ModelLoader,
        predict_eegnet,
        predict_mirepnet,
        EEGNET_CLASS_LABELS,
        MIREPNET_CLASS_LABELS
    )
except ImportError:
    # Fallback for direct script execution
    from utils import (
        detect_eeg_channels,
        extract_fp1_fp2,
        load_eeg_data,
        synchronize_data,
        validate_data_shape
    )
    from preprocessing import (
        preprocess_for_eegnet,
        preprocess_for_mirepnet,
        resample_data,
        apply_bandpass_filter,
        apply_common_average_reference,
        normalize_data
    )
    from models_integration import (
        ModelLoader,
        predict_eegnet,
        predict_mirepnet,
        EEGNET_CLASS_LABELS,
        MIREPNET_CLASS_LABELS
    )


class UnifiedEEGPipeline:
    """
    Unified pipeline for processing EEG data and running both models.
    """
    
    def __init__(self,
                 eegnet_model_path: Optional[str] = None,
                 mirepnet_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the unified pipeline.
        
        Parameters:
        -----------
        eegnet_model_path : Optional[str]
            Path to EEGNet model file
        mirepnet_model_path : Optional[str]
            Path to MIRepNet model file
        device : Optional[str]
            Device for PyTorch model ('cuda' or 'cpu')
        """
        self.model_loader = ModelLoader(
            eegnet_path=eegnet_model_path,
            mirepnet_path=mirepnet_model_path,
            device=device
        )
        
        # Configuration
        self.eegnet_config = {
            'target_fs': 250.0,
            'window_size': 250,
            'normalize': True
        }
        
        self.mirepnet_config = {
            'target_fs': 128.0,
            'lowcut': 4.0,
            'highcut': 38.0,
            'target_length': 512,
            'apply_car': True,
            'normalize': True
        }
    
    def load_data(self, 
                  file_path: Union[str, Path],
                  file_type: Optional[str] = None) -> Tuple[np.ndarray, List[str], float]:
        """
        Load EEG data from file.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            Path to EEG data file
        file_type : Optional[str]
            File type hint ('csv', 'edf', 'fif', 'npy')
            
        Returns:
        --------
        Tuple[np.ndarray, List[str], float]
            - data: EEG data array
            - channel_names: List of channel names
            - sampling_rate: Sampling frequency in Hz
        """
        return load_eeg_data(file_path, file_type)
    
    def detect_channels(self, 
                       data: np.ndarray,
                       channel_names: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Auto-detect EEG channels from data.
        
        Parameters:
        -----------
        data : np.ndarray
            EEG data array
        channel_names : Optional[List[str]]
            List of channel names if available
            
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping channel names to indices
        """
        return detect_eeg_channels(data, channel_names)
    
    def extract_emergency_copy(self, 
                               data: np.ndarray,
                               channel_map: Dict[str, int]) -> np.ndarray:
        """
        Extract emergency copy using FP1 and FP2 channels.
        
        Parameters:
        -----------
        data : np.ndarray
            Full EEG data
        channel_map : Dict[str, int]
            Dictionary mapping channel names to indices
            
        Returns:
        --------
        np.ndarray
            Extracted FP1/FP2 data
        """
        return extract_fp1_fp2(data, channel_map)
    
    def preprocess_for_eegnet(self,
                              fp_data: np.ndarray,
                              fs: float) -> np.ndarray:
        """
        Preprocess FP1/FP2 data for EEGNet model.
        
        Parameters:
        -----------
        fp_data : np.ndarray
            FP1/FP2 data. Shape: (n_samples, 2) or (2, n_samples)
        fs : float
            Sampling frequency
            
        Returns:
        --------
        np.ndarray
            Preprocessed data. Shape: (n_windows, 250, 2)
        """
        return preprocess_for_eegnet(
            fp_data,
            fs,
            target_fs=self.eegnet_config['target_fs'],
            window_size=self.eegnet_config['window_size'],
            normalize=self.eegnet_config['normalize']
        )
    
    def preprocess_for_mirepnet(self,
                                data: np.ndarray,
                                fs: float) -> np.ndarray:
        """
        Preprocess full EEG data for MIRepNet model.
        
        Parameters:
        -----------
        data : np.ndarray
            Full EEG data. Shape: (n_channels, n_samples) or (n_samples, n_channels)
        fs : float
            Sampling frequency
            
        Returns:
        --------
        np.ndarray
            Preprocessed data. Shape: (n_channels, 512)
        """
        return preprocess_for_mirepnet(
            data,
            fs,
            target_fs=self.mirepnet_config['target_fs'],
            lowcut=self.mirepnet_config['lowcut'],
            highcut=self.mirepnet_config['highcut'],
            target_length=self.mirepnet_config['target_length'],
            apply_car=self.mirepnet_config['apply_car'],
            normalize=self.mirepnet_config['normalize']
        )
    
    def run_eegnet(self,
                   preprocessed_data: np.ndarray,
                   threshold: float = 0.5) -> Dict:
        """
        Run EEGNet model inference.
        
        Parameters:
        -----------
        preprocessed_data : np.ndarray
            Preprocessed data. Shape: (n_windows, 250, 2) or (1, 250, 2)
        threshold : float
            Classification threshold (default: 0.5)
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - 'probabilities': Raw probabilities
            - 'predictions': Binary predictions
            - 'labels': Human-readable labels
            - 'summary': Summary statistics
        """
        if self.model_loader.eegnet_model is None:
            raise ValueError("EEGNet model not loaded")
        
        probabilities, predictions = predict_eegnet(
            self.model_loader.eegnet_model,
            preprocessed_data,
            threshold=threshold
        )
        
        # Convert to labels
        labels = [EEGNET_CLASS_LABELS[int(pred)] for pred in predictions]
        
        # Summary statistics
        n_blinks = np.sum(predictions == 1)
        n_no_blinks = np.sum(predictions == 0)
        avg_prob = np.mean(probabilities)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'labels': labels,
            'summary': {
                'n_blinks': int(n_blinks),
                'n_no_blinks': int(n_no_blinks),
                'avg_probability': float(avg_prob),
                'blink_rate': float(n_blinks / len(predictions)) if len(predictions) > 0 else 0.0
            }
        }
    
    def run_mirepnet(self,
                    preprocessed_data: np.ndarray) -> Dict:
        """
        Run MIRepNet model inference.
        
        Parameters:
        -----------
        preprocessed_data : np.ndarray
            Preprocessed data. Shape: (n_channels, 512) or (1, n_channels, 512)
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - 'probabilities': Class probabilities
            - 'predictions': Predicted class indices
            - 'labels': Human-readable labels
            - 'confidence': Confidence score
        """
        if self.model_loader.mirepnet_model is None:
            raise ValueError("MIRepNet model not loaded")
        
        probabilities, predictions = predict_mirepnet(
            self.model_loader.mirepnet_model,
            preprocessed_data,
            device=self.model_loader.device,
            return_probs=True
        )
        
        # Get predicted class
        pred_class = int(predictions[0])
        pred_label = MIREPNET_CLASS_LABELS.get(pred_class, f'Unknown_{pred_class}')
        confidence = float(probabilities[0, pred_class])
        
        # All class probabilities
        class_probs = {
            MIREPNET_CLASS_LABELS[i]: float(probabilities[0, i])
            for i in range(len(MIREPNET_CLASS_LABELS))
        }
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'label': pred_label,
            'class_index': pred_class,
            'confidence': confidence,
            'class_probabilities': class_probs
        }
    
    def process_file(self,
                     file_path: Union[str, Path],
                     file_type: Optional[str] = None,
                     eegnet_threshold: float = 0.5,
                     run_eegnet: bool = True,
                     run_mirepnet: bool = True) -> Dict:
        """
        Complete end-to-end processing of an EEG file.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            Path to EEG data file
        file_type : Optional[str]
            File type hint
        eegnet_threshold : float
            Threshold for EEGNet classification (default: 0.5)
        run_eegnet : bool
            Whether to run EEGNet model (default: True)
        run_mirepnet : bool
            Whether to run MIRepNet model (default: True)
            
        Returns:
        --------
        Dict
            Dictionary containing results from both models
        """
        results = {
            'file_path': str(file_path),
            'eegnet_results': None,
            'mirepnet_results': None,
            'errors': []
        }
        
        try:
            # Step 1: Load data
            print(f"📂 Loading data from {file_path}...")
            data, channel_names, fs = self.load_data(file_path, file_type)
            print(f"   Data shape: {data.shape}, Sampling rate: {fs} Hz")
            print(f"   Channels: {len(channel_names)}")
            
            # Step 2: Detect channels
            print("🔍 Detecting EEG channels...")
            channel_map = self.detect_channels(data, channel_names)
            print(f"   Detected channels: {list(channel_map.keys())}")
            
            # Step 3: Extract emergency copy (FP1/FP2)
            if run_eegnet:
                print("📋 Extracting emergency copy (FP1/FP2)...")
                try:
                    fp_data = self.extract_emergency_copy(data, channel_map)
                    print(f"   FP1/FP2 data shape: {fp_data.shape}")
                except ValueError as e:
                    results['errors'].append(f"FP1/FP2 extraction failed: {e}")
                    run_eegnet = False
            
            # Step 4: Preprocess for EEGNet
            if run_eegnet:
                print("⚙️  Preprocessing for EEGNet...")
                try:
                    eegnet_data = self.preprocess_for_eegnet(fp_data, fs)
                    print(f"   Preprocessed shape: {eegnet_data.shape}")
                except Exception as e:
                    results['errors'].append(f"EEGNet preprocessing failed: {e}")
                    run_eegnet = False
            
            # Step 5: Preprocess for MIRepNet
            if run_mirepnet:
                print("⚙️  Preprocessing for MIRepNet...")
                try:
                    mirepnet_data = self.preprocess_for_mirepnet(data, fs)
                    print(f"   Preprocessed shape: {mirepnet_data.shape}")
                except Exception as e:
                    results['errors'].append(f"MIRepNet preprocessing failed: {e}")
                    run_mirepnet = False
            
            # Step 6: Run EEGNet
            if run_eegnet and self.model_loader.eegnet_model is not None:
                print("🧠 Running EEGNet model...")
                try:
                    eegnet_results = self.run_eegnet(eegnet_data, threshold=eegnet_threshold)
                    results['eegnet_results'] = eegnet_results
                    print(f"   ✅ EEGNet completed: {eegnet_results['summary']['n_blinks']} blinks detected")
                except Exception as e:
                    results['errors'].append(f"EEGNet inference failed: {e}")
            
            # Step 7: Run MIRepNet
            if run_mirepnet and self.model_loader.mirepnet_model is not None:
                print("🧠 Running MIRepNet model...")
                try:
                    mirepnet_results = self.run_mirepnet(mirepnet_data)
                    results['mirepnet_results'] = mirepnet_results
                    print(f"   ✅ MIRepNet completed: {mirepnet_results['label']} (confidence: {mirepnet_results['confidence']:.2f})")
                except Exception as e:
                    results['errors'].append(f"MIRepNet inference failed: {e}")
            
            print("✅ Processing complete!")
            
        except Exception as e:
            results['errors'].append(f"Pipeline error: {str(e)}")
            print(f"❌ Error: {e}")
        
        return results


def create_pipeline(eegnet_model_path: Optional[str] = None,
                   mirepnet_model_path: Optional[str] = None,
                   device: Optional[str] = None) -> UnifiedEEGPipeline:
    """
    Convenience function to create a unified pipeline.
    
    Parameters:
    -----------
    eegnet_model_path : Optional[str]
        Path to EEGNet model file
    mirepnet_model_path : Optional[str]
        Path to MIRepNet model file
    device : Optional[str]
        Device for PyTorch model
        
    Returns:
    --------
    UnifiedEEGPipeline
        Initialized pipeline
    """
    return UnifiedEEGPipeline(
        eegnet_model_path=eegnet_model_path,
        mirepnet_model_path=mirepnet_model_path,
        device=device
    )

