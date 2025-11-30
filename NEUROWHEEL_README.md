# 🧠 NeuroWheel – Real-time Brain-Controlled Wheelchair Simulator

**THE MOST INSANE BCI WHEELCHAIR DEMO EVER BUILT**

A jaw-dropping 3D Streamlit web application that moves a wheelchair in real-time based on EEG Motor Imagery predictions. When you see this, your reaction will be: **"WHAT THE HELL?! THE WHEELCHAIR IS MOVING WITH HIS BRAIN!"**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

**Windows:**
```bash
run_neurowheel.bat
```

**Or directly:**
```bash
streamlit run app.py --server.maxUploadSize=500
```

### 3. Upload Your EDF File

1. Open the app in your browser (usually `http://localhost:8501`)
2. Click "DROP YOUR .EDF FILE HERE"
3. Upload an EDF file
4. Click "🚀 START SIMULATION"
5. **WATCH THE MAGIC HAPPEN!**

## ✨ Features

### 🎨 Visual Design
- **Cyberpunk/Tron Legacy aesthetic**
- Neon green trail (#39FF14) that glows behind the wheelchair
- Dark room with grid floor
- Smooth 3D wheelchair model with spinning wheels
- Third-person camera that follows the wheelchair
- Real-time physics-based movement

### 🧠 Brain Control
- **EEGNet**: Detects startle blinks in real-time
- **MIRepNet**: Classifies motor imagery (Forward, Left, Right, Stop)
- Processes EDF files in 1-second chunks
- True online simulation (not pre-recorded)

### 📊 Live HUD
- **Current Command**: Huge neon display showing current brain command
- **Confidence Meter**: Visual confidence bar
- **Blink Counter**: Real-time blink detection count
- **Speedometer**: Simulated speed in km/h
- **Distance Traveled**: Total distance in meters
- **Timer**: Elapsed time and progress
- **Command History**: Last 10 commands with timestamps

### 🎮 Controls
- **Reset Path**: Clear the trail and reset position
- **Export Results**: Download CSV with all commands and timestamps

## 🎯 How It Works

1. **Upload EDF File**: User uploads an EEG recording
2. **Load Models**: App loads both EEGNet and MIRepNet models
3. **Process in Chunks**: EDF file is processed in 1-second windows
4. **Extract FP1/FP2**: Emergency copy channels extracted automatically
5. **Run Models**:
   - EEGNet detects blinks (FP1/FP2, 250 samples at 250 Hz)
   - MIRepNet classifies motor imagery (64 channels, 512 samples at 128 Hz)
6. **Update Physics**: Wheelchair position/rotation updated based on command
7. **Render 3D**: Plotly 3D scene updates in real-time with smooth animation

## 🎨 Command Mapping

- **Forward** (Both Fists): Wheelchair accelerates forward
- **Left** (Left Fist): Rotates left and moves
- **Right** (Right Fist): Rotates right and moves
- **Stop** (Both Feet): Decelerates smoothly to stop, trail turns gray

## 📁 File Structure

```
.
├── app.py                    # Main Streamlit application
├── run_neurowheel.bat        # Windows launcher script
├── requirements.txt          # Python dependencies
├── src/                      # Pipeline modules
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── models_integration.py
│   └── utils.py
└── NEUROWHEEL_README.md      # This file
```

## 🔧 Technical Details

### Models Used
- **EEGNet**: `startle_blink_EEGNet_99_attention.keras`
- **MIRepNet**: `mirapnet_final_model.pth`

### Preprocessing
- **EEGNet**: 250 samples at 250 Hz, Z-score normalization
- **MIRepNet**: 512 samples at 128 Hz, 4-38 Hz bandpass, CAR, normalization

### 3D Rendering
- **Plotly 3D**: Smooth, interactive 3D visualization
- **Real-time updates**: Scene refreshes as predictions come in
- **Physics simulation**: Acceleration, deceleration, rotation

## 🎬 Example Output

When running, you'll see:
- A 3D wheelchair moving smoothly in a dark room
- A glowing neon-green trail following the wheelchair
- Real-time command updates in the sidebar
- Smooth camera following the wheelchair
- Blink counter and confidence meters updating live

## 🐛 Troubleshooting

### "Models not loading"
- Check that model paths in `app.py` are correct
- Ensure model files exist in the specified locations

### "EDF file not loading"
- Verify the file is a valid EDF format
- Check that the file contains EEG channels (especially FP1/FP2)

### "3D scene not updating"
- Make sure simulation is running (click "START SIMULATION")
- Check browser console for errors
- Try refreshing the page

### Performance Issues
- Reduce chunk size in `process_edf_chunk()` if processing is slow
- Use GPU if available (change `device='cpu'` to `device='cuda'`)

## 🎉 Credits

Built with:
- **Streamlit**: Web framework
- **Plotly**: 3D visualization
- **MNE-Python**: EEG data handling
- **TensorFlow/Keras**: EEGNet model
- **PyTorch**: MIRepNet model

## 📝 License

This is a student project for demonstration purposes.

---

**NOW GO RUN IT AND BE AMAZED! 🚀🧠✨**

