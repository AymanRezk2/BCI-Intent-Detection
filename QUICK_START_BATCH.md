# Quick Start: Batch Processing EDF Files

## 🚀 Fast Setup (3 Steps)

### 1. Update Configuration

Edit `batch_process_edf.py` and update these lines (around line 350):

```python
EEGNET_MODEL_PATH = "startle_blink_EEGNet_99_attention.keras"
MIREPNET_MODEL_PATH = "mirapnet_final_model.pth"
DATA_FOLDER = r"D:\DEPI\Finel\data\files"  # ← Your data folder
OUTPUT_CSV = "batch_processing_results.csv"
DEVICE = 'cpu'  # or 'cuda' if you have GPU
```

### 2. Run the Script

```bash
python batch_process_edf.py
```

### 3. Check Results

Results are saved to `batch_processing_results.csv` and displayed in the console.

## 📁 Expected Folder Structure

```
D:\DEPI\Finel\data\files\
├── S001\
│   ├── S001R01.edf
│   ├── S001R01.edf.event  (optional)
│   └── S001R02.edf
├── S002\
│   └── ...
```

## 📊 What You Get

- **Console output**: Real-time progress for each file
- **CSV file**: Complete results with:
  - Number of blinks detected (EEGNet)
  - Motor imagery classification (MIRepNet)
  - Confidence scores
  - Error information

## 🔧 Alternative: Simple Example

For a simpler approach, use `batch_process_example.py`:

```python
# Edit batch_process_example.py, then run:
python batch_process_example.py
```

## ⚡ Key Features

✅ Automatically finds all `.edf` files  
✅ Handles missing `.event` files gracefully  
✅ Processes through both EEGNet and MIRepNet  
✅ Extracts FP1/FP2 automatically  
✅ Saves results to CSV  
✅ Shows progress and summary statistics  

## 📝 Example Output

```
📊 Results for S001R01.edf:
   🧠 EEGNet (Startle Blink Detection):
      - Blinks detected: 12
      - Blink rate: 5.88%
   🧠 MIRepNet (Motor Imagery Classification):
      - Predicted class: Forward
      - Confidence: 85.00%
```

## 🆘 Troubleshooting

**No files found?**  
→ Check that `DATA_FOLDER` path is correct (use `r"..."` for Windows paths)

**Model errors?**  
→ Ensure model files are in the same directory as the script

**Memory issues?**  
→ Process files in smaller batches or use `device='cpu'`

For more details, see `BATCH_PROCESSING_README.md`

