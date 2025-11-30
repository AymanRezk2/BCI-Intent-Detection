"""
Simple example of using the batch processing script.

This is a minimal example showing how to process EDF files in batch.
"""

from batch_process_edf import EDFBatchProcessor

# ============================================================================
# Configuration
# ============================================================================

# Update these paths to match your setup
EEGNET_MODEL = "startle_blink_EEGNet_99_attention.keras"
MIREPNET_MODEL = "mirapnet_final_model.pth"
DATA_FOLDER = r"D:\DEPI\Finel\data\files"  # Update this path
OUTPUT_FILE = "results.csv"

# ============================================================================
# Process Files
# ============================================================================

# Create processor
processor = EDFBatchProcessor(
    eegnet_model_path=EEGNET_MODEL,
    mirepnet_model_path=MIREPNET_MODEL,
    device='cpu'  # Change to 'cuda' if you have GPU
)

# Process all EDF files in the folder
results_df = processor.process_folder(
    root_folder=DATA_FOLDER,
    output_csv=OUTPUT_FILE
)

# Display results
print("\n" + "="*70)
print("📊 Processing Complete!")
print("="*70)
print(f"\nProcessed {len(results_df)} files")
print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nFirst few results:")
print(results_df.head())

