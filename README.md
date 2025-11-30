# Unified EEG Processing Pipeline

A complete end-to-end pipeline for processing EEG signals and running both **EEGNet** (startle blink detection) and **MIRepNet** (motor imagery classification) models.

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── utils.py                 # Channel detection, data loading utilities
│   ├── preprocessing.py         # Unified preprocessing functions
│   ├── models_integration.py   # Model loading and inference
│   └── pipeline.py              # Main end-to-end pipeline
├── example_usage.py             # Usage examples
├── README.md                    # This file
├── startle_blink_EEGNet_99_attention.keras  # EEGNet model
└── mirapnet_final_model.pth     # MIRepNet model
```

## 🎯 Features

- **Auto-detection of EEG channels**: Dynamically detects FP1, FP2, and other channels
- **Unified preprocessing**: Shared preprocessing pipeline for both models
- **Emergency copy extraction**: Automatically extracts FP1/FP2 for blink detection
- **Dual model support**: Runs both EEGNet and MIRepNet models
- **Flexible data loading**: Supports CSV, EDF, FIF, and NumPy formats
- **Production-ready**: Modular, clean, and well-documented code

## 📋 Requirements

```bash
pip install numpy scipy tensorflow torch mne
```

## 🚀 Quick Start

### Basic Usage

```python
from src.pipeline import create_pipeline

# Create pipeline with both models
pipeline = create_pipeline(
    eegnet_model_path="startle_blink_EEGNet_99_attention.keras",
    mirepnet_model_path="mirapnet_final_model.pth",
    device='cpu'  # or 'cuda' for GPU
)

# Process an EEG file
results = pipeline.process_file(
    file_path="path/to/your/eeg_data.csv",
    eegnet_threshold=0.5,
    run_eegnet=True,
    run_mirepnet=True
)

# Access results
if results['eegnet_results']:
    print(f"Blinks detected: {results['eegnet_results']['summary']['n_blinks']}")

if results['mirepnet_results']:
    print(f"MI Class: {results['mirepnet_results']['label']}")
```

### Step-by-Step Processing

```python
# Load data
data, channel_names, fs = pipeline.load_data("eeg_data.csv")

# Detect channels
channel_map = pipeline.detect_channels(data, channel_names)

# Extract FP1/FP2
fp_data = pipeline.extract_emergency_copy(data, channel_map)

# Preprocess
eegnet_data = pipeline.preprocess_for_eegnet(fp_data, fs)
mirepnet_data = pipeline.preprocess_for_mirepnet(data, fs)

# Run models
eegnet_results = pipeline.run_eegnet(eegnet_data)
mirepnet_results = pipeline.run_mirepnet(mirepnet_data)
```

## 📊 Model Details

### EEGNet (Startle Blink Detection)

- **Input**: 2 channels (FP1, FP2), 250 samples (1 second at 250 Hz)
- **Output**: Binary classification (blink/no blink)
- **Preprocessing**:
  - Resample to 250 Hz
  - Z-score normalization
  - Window size: 250 samples

### MIRepNet (Motor Imagery Classification)

- **Input**: 64 channels, 512 timesteps (4 seconds at 128 Hz)
- **Output**: 4 classes (Stop, Left, Right, Forward)
- **Preprocessing**:
  - Bandpass filter: 4-38 Hz
  - Resample to 128 Hz
  - Common Average Reference (CAR)
  - Pad/truncate to 512 timesteps
  - Z-score normalization

## 🔧 Configuration

You can customize preprocessing parameters:

```python
pipeline = create_pipeline(...)

# Modify EEGNet config
pipeline.eegnet_config['target_fs'] = 250.0
pipeline.eegnet_config['window_size'] = 250
pipeline.eegnet_config['normalize'] = True

# Modify MIRepNet config
pipeline.mirepnet_config['target_fs'] = 128.0
pipeline.mirepnet_config['lowcut'] = 4.0
pipeline.mirepnet_config['highcut'] = 38.0
pipeline.mirepnet_config['target_length'] = 512
pipeline.mirepnet_config['apply_car'] = True
pipeline.mirepnet_config['normalize'] = True
```

## 📝 Supported File Formats

- **CSV**: Comma-separated values (time, channel1, channel2, ...)
- **EDF**: European Data Format (via MNE)
- **FIF**: MNE-Python format
- **NPY**: NumPy array format

## 🔍 Channel Detection

The pipeline automatically detects EEG channels using:
- Channel names (if provided): Matches FP1, FP2, Fp1, Fp2, etc.
- Data shape: Infers channels from array dimensions
- Flexible naming: Handles various naming conventions

## ⚠️ Important Notes

1. **FP1/FP2 Detection**: The pipeline automatically detects and extracts FP1/FP2 channels. If these channels are not found, EEGNet will not run.

2. **Data Synchronization**: The pipeline ensures all data is synchronized before processing.

3. **Model Compatibility**: 
   - EEGNet expects exactly 2 channels (FP1, FP2)
   - MIRepNet expects 64 channels (will work with fewer, but performance may vary)

4. **Memory Usage**: For large files, consider processing in chunks or windows.

## 🐛 Troubleshooting

### "FP1/FP2 channels not found"
- Ensure your data contains FP1 and FP2 channels
- Check channel names match expected patterns
- Verify data shape is correct

### "Model loading failed"
- Check model file paths are correct
- Ensure TensorFlow is installed for EEGNet
- Ensure PyTorch is installed for MIRepNet

### "Shape mismatch errors"
- Verify data dimensions match expected shapes
- Check sampling frequency matches model requirements
- Ensure preprocessing steps are applied correctly

## 📚 API Reference

See the docstrings in each module for detailed API documentation:
- `src/utils.py`: Channel detection and data loading
- `src/preprocessing.py`: Preprocessing functions
- `src/models_integration.py`: Model loading and inference
- `src/pipeline.py`: Main pipeline class

## 🤝 Contributing

This pipeline was created to unify preprocessing and inference for both EEGNet and MIRepNet models. The code is modular and can be easily extended for additional models or preprocessing steps.

## 📄 License

This code is provided as-is for research and development purposes.

