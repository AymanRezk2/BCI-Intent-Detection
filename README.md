# NeuroWheel - Brain-Controlled Wheelchair

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-EEG-orange.svg)]()



> Control powered wheelchairs through brain signals and eye blinks. Empowering individuals with motor disabilities through hands-free navigation.

## Overview

**NeuroWheel** uses EEG brain signals to control wheelchairs for individuals with motor disabilities. The system combines two deep learning models:

- **Motor Imagery (MIRepNet)**: Think "left", "right", "forward", or "stop" to navigate
- **Blink Detection (EEGNet)**: Voluntary eye blinks trigger emergency stop (99% accuracy)

## 🔴 Live Demo

Experience NeuroWheel in action through our interactive web application:
👉 **Streamlit App:** https://depi-bci-27.streamlit.app/

This demo showcases:
- EEG signal preprocessing
- Motor imagery direction prediction
- Blink-based emergency stop simulation
> ⚠️ Note: This demo is for research and educational purposes and does not replace medical-certified hardware systems.


### Target Users

Individuals with spinal cord injuries, ALS, quadriplegia, or severe motor impairments.

## Key Features

- **4-Direction Control**: Left, Right, Forward, Stop via thoughts
- **Emergency Stop**: Blink-based safety mechanism
- **Real-time Processing**: Low-latency for safe navigation
- **Auto-Channel Detection**: Works with standard EEG caps
- **Multiple Formats**: CSV, EDF, FIF, NumPy support

## Installation

```bash
git clone https://github.com/AymanRezk2/BCI-Intent-Detection.git
cd BCI-Intent-Detection

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: `numpy`, `scipy`, `tensorflow`, `torch`, `mne`

## System Architecture

### EEGNet - Emergency Stop

- **Input**: FP1, FP2 channels (frontal electrodes)
- **Output**: Blink detected / No blink
- **Accuracy**: 99%+

### MIRepNet - Directional Control

- **Input**: 64 EEG channels
- **Output**: 4 classes (Stop, Left, Right, Forward)
- **Method**: Motor imagery classification

## Project Structure

```
BCI-Intent-Detection/
├── experiments/
│   ├── CNN/
│   ├── EEGNet/
│   ├── P300/
│   └── MIRepNet.ipynb
├── models/
│   ├── mirapnet_final_model.pth
│   └── startle_blink_EEGNet_99_attention.keras
├── notebooks/
│   ├── EEGNT_1.ipynb
│   └── MIRepNetFinal.ipynb
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── batch_process_edf.py
│   ├── batch_process_example.py
│   ├── example_usage.py
│   ├── models_integration.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── utils.py
├── app.py
├── BCI_Project_Proposal.docx
├── NEUROWHEEL_Presentation.pdf
├── README.md
├── requirements.txt
└── run_neurowheel.bat
```

## Model Details

### Preprocessing Pipeline

**EEGNet (Blink Detection)**:

- Resample to 250 Hz
- Z-score normalization
- Window: 250 samples (1 second)

**MIRepNet (Motor Imagery)**:

- Bandpass filter: 4-38 Hz
- Resample to 128 Hz
- Common Average Reference
- Window: 512 timesteps (4 seconds)

## Configuration

```python
# Customize preprocessing
pipeline.eegnet_config['target_fs'] = 250.0
pipeline.mirepnet_config['lowcut'] = 4.0
pipeline.mirepnet_config['highcut'] = 38.0
```

## Supported Formats

- **CSV**: Time series data with channel columns
- **EDF**: European Data Format
- **FIF**: MNE-Python format
- **NPY**: NumPy arrays

## Important Notes

- **FP1/FP2 Required**: Frontal electrodes needed for blink detection
- **64 Channels Recommended**: For optimal motor imagery performance
- **GPU Recommended**: For real-time processing

## Troubleshooting

**"FP1/FP2 not found"**: Check channel names (FP1, Fp1, FP_1 variants supported)

**"Model loading failed"**: Verify TensorFlow and PyTorch are installed

**Memory errors**: Process in smaller chunks or use CPU

## Documentation

- **Project Proposal**: `BCI_Project_Proposal.docx`
- **Presentation**: `NEUROWHEEL_Presentation.pdf`
- **Notebooks**: See `notebooks/` for training details

## Roadmap

**Completed**:

- [x] EEGNet blink detection (99% accuracy)
- [x] MIRepNet 4-direction control
- [x] Unified preprocessing pipeline
- [x] Real-time streaming interface
- [x] Web-based control dashboard

**Future**:

- [ ] Mobile app for monitoring
- [ ] User calibration module
- [ ] Clinical trial deployment
- [ ] Hardware integration (Arduino/RPi)

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 📧 Contact

**Ayman Rezk** - [@AymanRezk2](https://github.com/AymanRezk2)

Project Link: [https://github.com/AymanRezk2/BCI-Intent-Detection](https://github.com/AymanRezk2/BCI-Intent-Detection)

---

⭐ **Star this repo** if you find it helpful!

