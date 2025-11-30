# Batch Processing EDF Files

This guide explains how to use the batch processing script to process multiple EDF files through the unified EEG pipeline.

## 📁 File Structure

The batch processor expects a folder structure like:

```
D:\DEPI\Finel\data\files\
├── S001\
│   ├── S001R01.edf
│   ├── S001R01.edf.event  (optional)
│   ├── S001R02.edf
│   └── S001R02.edf.event  (optional)
├── S002\
│   ├── S002R01.edf
│   └── ...
```

## 🚀 Quick Start

### Option 1: Use the Main Script

```bash
python batch_process_edf.py
```

**Before running**, update the configuration in `batch_process_edf.py`:

```python
# Update these paths
EEGNET_MODEL_PATH = "startle_blink_EEGNet_99_attention.keras"
MIREPNET_MODEL_PATH = "mirapnet_final_model.pth"
DATA_FOLDER = r"D:\DEPI\Finel\data\files"
OUTPUT_CSV = "batch_processing_results.csv"
DEVICE = 'cpu'  # or 'cuda'
```

### Option 2: Use the Simple Example

```bash
python batch_process_example.py
```

Update the paths in `batch_process_example.py` first.

### Option 3: Use in Your Own Script

```python
from batch_process_edf import EDFBatchProcessor

# Initialize
processor = EDFBatchProcessor(
    eegnet_model_path="startle_blink_EEGNet_99_attention.keras",
    mirepnet_model_path="mirapnet_final_model.pth",
    device='cpu'
)

# Process folder
results_df = processor.process_folder(
    root_folder=r"D:\DEPI\Finel\data\files",
    output_csv="results.csv"
)
```

## 📊 Output

The script generates:

1. **Console Output**: Real-time progress and results for each file
2. **CSV File**: Complete results in tabular format

### CSV Columns

- `file_path`: Full path to the EDF file
- `filename`: EDF filename
- `subject`: Subject folder name (e.g., 'S001')
- `timestamp`: Processing timestamp
- `status`: Processing status ('success', 'error', 'no_results')
- `eegnet_n_blinks`: Number of blinks detected
- `eegnet_blink_rate`: Blink detection rate
- `mirepnet_label`: Predicted motor imagery class
- `mirepnet_confidence`: Confidence score
- `mirepnet_prob_forward`, `mirepnet_prob_left`, etc.: Class probabilities
- `has_event_file`: Whether .event file was found
- `errors`: Any errors encountered

## 🔧 Features

- ✅ **Automatic file discovery**: Recursively finds all .edf files
- ✅ **Optional event files**: Gracefully handles missing .event files
- ✅ **Progress tracking**: Shows progress for each file
- ✅ **Error handling**: Continues processing even if some files fail
- ✅ **Summary statistics**: Overall statistics at the end
- ✅ **CSV export**: Easy-to-analyze results in CSV format

## 📝 Example Output

```
🚀 EEG Batch Processing Pipeline
======================================================================
📂 Scanning for EDF files in: D:\DEPI\Finel\data\files
   Found 5 EDF file(s)

🔄 Processing 5 file(s)...

[1/5] Processing S001R01.edf...
======================================================================
📄 Processing: S001R01.edf
======================================================================
📂 Loading data from S001R01.edf...
   Data shape: (64, 51200), Sampling rate: 128.0 Hz
   Channels: 64
🔍 Detecting EEG channels...
   Detected channels: ['FP1', 'FP2', 'F3', 'F4', ...]
📋 Extracting emergency copy (FP1/FP2)...
   FP1/FP2 data shape: (2, 51200)
⚙️  Preprocessing for EEGNet...
   Preprocessed shape: (204, 250, 2)
⚙️  Preprocessing for MIRepNet...
   Preprocessed shape: (64, 512)
🧠 Running EEGNet model...
   ✅ EEGNet completed: 12 blinks detected
🧠 Running MIRepNet model...
   ✅ MIRepNet completed: Forward (confidence: 0.85)

📊 Results for S001R01.edf:
   🧠 EEGNet (Startle Blink Detection):
      - Blinks detected: 12
      - No blinks: 192
      - Blink rate: 5.88%
      - Avg probability: 0.523
   🧠 MIRepNet (Motor Imagery Classification):
      - Predicted class: Forward
      - Confidence: 85.00%
      - Class probabilities:
        • Forward: 85.00%
        • Left: 8.00%
        • Right: 5.00%
        • Stop: 2.00%
   ✅ Status: success

======================================================================
📊 OVERALL SUMMARY
======================================================================
Total files processed: 5
  ✅ Successful: 5
  ❌ Errors: 0

🧠 EEGNet Summary:
  - Total blinks detected: 45
  - Average blink rate: 5.23%

🧠 MIRepNet Summary:
  - Forward: 2 (40.0%)
  - Left: 1 (20.0%)
  - Right: 1 (20.0%)
  - Stop: 1 (20.0%)
======================================================================
```

## ⚠️ Troubleshooting

### "No EDF files found"
- Check that `DATA_FOLDER` path is correct
- Ensure the path uses Windows format: `r"D:\path\to\folder"`
- Verify EDF files have `.edf` extension

### "Model loading failed"
- Check model file paths are correct
- Ensure model files exist in the working directory
- Verify TensorFlow/PyTorch are installed

### "FP1/FP2 channels not found"
- Some EDF files may not have FP1/FP2 channels
- The script will continue processing other files
- Check channel names in your EDF files

### Memory Issues
- Process files in smaller batches
- Use `device='cpu'` if GPU memory is limited
- Close other applications to free memory

## 🔍 Advanced Usage

### Process Specific Subject Folders

```python
# Process only S001 folder
processor = EDFBatchProcessor(...)
results = processor.process_folder(
    root_folder=r"D:\DEPI\Finel\data\files\S001",
    output_csv="S001_results.csv"
)
```

### Access Individual Results

```python
# Process and access results
results_df = processor.process_folder(...)

# Filter successful results
successful = results_df[results_df['status'] == 'success']

# Get files with blinks
with_blinks = results_df[results_df['eegnet_n_blinks'] > 0]

# Get specific MI class
forward_class = results_df[results_df['mirepnet_label'] == 'Forward']
```

### Custom Processing

```python
# Process files one by one
processor = EDFBatchProcessor(...)
edf_files = processor.find_edf_files(r"D:\DEPI\Finel\data\files")

for file_info in edf_files:
    result = processor.process_single_file(
        edf_path=file_info['edf_path'],
        event_path=file_info['event_path']
    )
    # Do something with result
    print(f"Processed {result['filename']}: {result['status']}")
```

## 📚 See Also

- `src/pipeline.py`: Main pipeline implementation
- `example_usage.py`: Single file processing examples
- `README.md`: General pipeline documentation

