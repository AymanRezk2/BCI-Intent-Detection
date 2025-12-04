"""
Example usage of the unified EEG processing pipeline.

This script demonstrates how to use the pipeline to process EEG data
and run both models (EEGNet and MIRepNet).
"""

from pathlib import Path
from src.pipeline import create_pipeline

# ============================================================================
# Configuration
# ============================================================================

# Paths to model files (adjust these to match your actual file locations)
EEGNET_MODEL_PATH = "D:\DEPI\finel depi\startle_blink_EEGNet_99_attention.keras"
MIREPNET_MODEL_PATH = "D:\DEPI\finel depi\mirapnet_final_model.pth"

# Path to your EEG data file (adjust this to your actual data file)
EEG_DATA_FILE = "path/to/your/eeg_data.csv"  # or .edf, .fif, .npy

# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """Basic example of using the pipeline."""
    
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    # Create pipeline with both models
    pipeline = create_pipeline(
        eegnet_model_path=EEGNET_MODEL_PATH,
        mirepnet_model_path=MIREPNET_MODEL_PATH,
        device='cpu'  # or 'cuda' if GPU is available
    )
    
    # Process a file
    results = pipeline.process_file(
        file_path=EEG_DATA_FILE,
        file_type=None,  # Auto-detect from extension
        eegnet_threshold=0.5,
        run_eegnet=True,
        run_mirepnet=True
    )
    
    # Print results
    print("\n📊 Results:")
    print(f"File: {results['file_path']}")
    
    if results['eegnet_results']:
        print("\n🧠 EEGNet Results (Startle Blink Detection):")
        summary = results['eegnet_results']['summary']
        print(f"  - Blinks detected: {summary['n_blinks']}")
        print(f"  - No blinks: {summary['n_no_blinks']}")
        print(f"  - Blink rate: {summary['blink_rate']:.2%}")
        print(f"  - Average probability: {summary['avg_probability']:.3f}")
    
    if results['mirepnet_results']:
        print("\n🧠 MIRepNet Results (Motor Imagery Classification):")
        mi_results = results['mirepnet_results']
        print(f"  - Predicted class: {mi_results['label']}")
        print(f"  - Confidence: {mi_results['confidence']:.2%}")
        print(f"  - All class probabilities:")
        for class_name, prob in mi_results['class_probabilities'].items():
            print(f"    - {class_name}: {prob:.2%}")
    
    if results['errors']:
        print("\n⚠️  Errors:")
        for error in results['errors']:
            print(f"  - {error}")


# ============================================================================
# Example 2: Step-by-Step Processing
# ============================================================================

def example_step_by_step():
    """Example showing step-by-step processing."""
    
    print("\n" + "=" * 70)
    print("Example 2: Step-by-Step Processing")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_pipeline(
        eegnet_model_path=EEGNET_MODEL_PATH,
        mirepnet_model_path=MIREPNET_MODEL_PATH
    )
    
    # Step 1: Load data
    data, channel_names, fs = pipeline.load_data(EEG_DATA_FILE)
    print(f"\n1. Loaded data: shape={data.shape}, fs={fs} Hz, channels={len(channel_names)}")
    
    # Step 2: Detect channels
    channel_map = pipeline.detect_channels(data, channel_names)
    print(f"\n2. Detected channels: {list(channel_map.keys())}")
    
    # Step 3: Extract FP1/FP2
    fp_data = pipeline.extract_emergency_copy(data, channel_map)
    print(f"\n3. Extracted FP1/FP2: shape={fp_data.shape}")
    
    # Step 4: Preprocess for EEGNet
    eegnet_data = pipeline.preprocess_for_eegnet(fp_data, fs)
    print(f"\n4. Preprocessed for EEGNet: shape={eegnet_data.shape}")
    
    # Step 5: Preprocess for MIRepNet
    mirepnet_data = pipeline.preprocess_for_mirepnet(data, fs)
    print(f"\n5. Preprocessed for MIRepNet: shape={mirepnet_data.shape}")
    
    # Step 6: Run models
    if pipeline.model_loader.eegnet_model:
        eegnet_results = pipeline.run_eegnet(eegnet_data)
        print(f"\n6. EEGNet results: {eegnet_results['summary']}")
    
    if pipeline.model_loader.mirepnet_model:
        mirepnet_results = pipeline.run_mirepnet(mirepnet_data)
        print(f"\n7. MIRepNet results: {mirepnet_results['label']} ({mirepnet_results['confidence']:.2%})")


# ============================================================================
# Example 3: Processing Multiple Files
# ============================================================================

def example_batch_processing():
    """Example of processing multiple files."""
    
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    
    # Create pipeline once
    pipeline = create_pipeline(
        eegnet_model_path=EEGNET_MODEL_PATH,
        mirepnet_model_path=MIREPNET_MODEL_PATH
    )
    
    # List of files to process
    data_files = [
        "path/to/file1.csv",
        "path/to/file2.edf",
        "path/to/file3.fif"
    ]
    
    all_results = []
    
    for file_path in data_files:
        if not Path(file_path).exists():
            print(f"⚠️  Skipping {file_path} (file not found)")
            continue
        
        print(f"\n📂 Processing {file_path}...")
        results = pipeline.process_file(file_path)
        all_results.append(results)
    
    # Summary
    print(f"\n✅ Processed {len(all_results)} files")
    for i, result in enumerate(all_results):
        print(f"\nFile {i+1}: {Path(result['file_path']).name}")
        if result['eegnet_results']:
            print(f"  Blinks: {result['eegnet_results']['summary']['n_blinks']}")
        if result['mirepnet_results']:
            print(f"  MI Class: {result['mirepnet_results']['label']}")


# ============================================================================
# Example 4: Custom Configuration
# ============================================================================

def example_custom_config():
    """Example with custom preprocessing configuration."""
    
    print("\n" + "=" * 70)
    print("Example 4: Custom Configuration")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_pipeline(
        eegnet_model_path=EEGNET_MODEL_PATH,
        mirepnet_model_path=MIREPNET_MODEL_PATH
    )
    
    # Modify configuration
    pipeline.eegnet_config['target_fs'] = 250.0
    pipeline.eegnet_config['window_size'] = 250
    pipeline.eegnet_config['normalize'] = True
    
    pipeline.mirepnet_config['target_fs'] = 128.0
    pipeline.mirepnet_config['lowcut'] = 4.0
    pipeline.mirepnet_config['highcut'] = 38.0
    pipeline.mirepnet_config['target_length'] = 512
    pipeline.mirepnet_config['apply_car'] = True
    pipeline.mirepnet_config['normalize'] = True
    
    # Process with custom config
    results = pipeline.process_file(EEG_DATA_FILE)
    print("✅ Processing complete with custom configuration")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("🚀 Unified EEG Processing Pipeline - Examples")
    print("=" * 70)
    print("\nNote: Update the file paths in this script before running!")
    print("\nUncomment the example you want to run:\n")
    
    # Uncomment the example you want to run:
    # example_basic_usage()
    # example_step_by_step()
    # example_batch_processing()
    # example_custom_config()
    
    print("\n💡 Tip: Uncomment one of the example functions above to run it.")

