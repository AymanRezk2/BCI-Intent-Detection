"""
Batch processing script for EEG EDF files.

This script iterates through EDF files in a folder structure, processes each file
through the unified pipeline (EEGNet + MIRepNet), and stores results.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline import create_pipeline


class EDFBatchProcessor:
    """
    Batch processor for EDF EEG files with optional event files.
    """
    
    def __init__(self,
                 eegnet_model_path: str,
                 mirepnet_model_path: str,
                 device: Optional[str] = None):
        """
        Initialize the batch processor.
        
        Parameters:
        -----------
        eegnet_model_path : str
            Path to EEGNet model file
        mirepnet_model_path : str
            Path to MIRepNet model file
        device : Optional[str]
            Device for PyTorch model ('cuda' or 'cpu')
        """
        print("🚀 Initializing unified pipeline...")
        self.pipeline = create_pipeline(
            eegnet_model_path=eegnet_model_path,
            mirepnet_model_path=mirepnet_model_path,
            device=device
        )
        self.results = []
    
    def find_edf_files(self, root_folder: str) -> List[Dict[str, Path]]:
        """
        Find all EDF files in the folder structure.
        
        Parameters:
        -----------
        root_folder : str
            Root folder containing subject folders (e.g., 'D:\\DEPI\\Finel\\data\\files')
            
        Returns:
        --------
        List[Dict[str, Path]]
            List of dictionaries with 'edf_path' and 'event_path' keys
        """
        root_path = Path(root_folder)
        edf_files = []
        
        print(f"\n📂 Scanning for EDF files in: {root_path}")
        
        # Recursively find all .edf files
        for edf_path in root_path.rglob("*.edf"):
            # Look for corresponding .event file
            event_path = edf_path.with_suffix('.edf.event')
            
            edf_files.append({
                'edf_path': edf_path,
                'event_path': event_path if event_path.exists() else None,
                'subject': edf_path.parent.name,  # e.g., 'S001'
                'filename': edf_path.name  # e.g., 'S001R01.edf'
            })
        
        print(f"   Found {len(edf_files)} EDF file(s)")
        return edf_files
    
    def read_event_file(self, event_path: Path) -> Optional[Dict]:
        """
        Read event file if it exists.
        
        Parameters:
        -----------
        event_path : Path
            Path to .event file
            
        Returns:
        --------
        Optional[Dict]
            Dictionary with event information, or None if file doesn't exist
        """
        if event_path is None or not event_path.exists():
            return None
        
        try:
            # Try to read as text file (handle different encodings)
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(event_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # Fallback: read as binary and decode with errors='ignore'
                with open(event_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
            
            # Parse events (adjust parsing logic based on your event file format)
            lines = content.splitlines()
            events = {
                'file_path': str(event_path),
                'content': content,
                'num_lines': len(lines),
                'non_empty_lines': len([l for l in lines if l.strip()])
            }
            
            return events
        except Exception as e:
            print(f"   ⚠️  Warning: Could not read event file {event_path}: {e}")
            return None
    
    def process_single_file(self, 
                            edf_path: Path,
                            event_path: Optional[Path] = None) -> Dict:
        """
        Process a single EDF file through the pipeline.
        
        Parameters:
        -----------
        edf_path : Path
            Path to EDF file
        event_path : Optional[Path]
            Path to optional event file
            
        Returns:
        --------
        Dict
            Results dictionary with processing results
        """
        result = {
            'file_path': str(edf_path),
            'filename': edf_path.name,
            'subject': edf_path.parent.name,
            'timestamp': datetime.now().isoformat(),
            'eegnet_results': None,
            'mirepnet_results': None,
            'event_info': None,
            'errors': [],
            'status': 'pending'
        }
        
        try:
            # Read event file if it exists
            if event_path and event_path.exists():
                result['event_info'] = self.read_event_file(event_path)
            
            # Process through pipeline
            print(f"\n{'='*70}")
            print(f"📄 Processing: {edf_path.name}")
            print(f"{'='*70}")
            
            pipeline_results = self.pipeline.process_file(
                file_path=str(edf_path),
                file_type='edf',
                eegnet_threshold=0.5,
                run_eegnet=True,
                run_mirepnet=True
            )
            
            # Extract results
            result['eegnet_results'] = pipeline_results.get('eegnet_results')
            result['mirepnet_results'] = pipeline_results.get('mirepnet_results')
            result['errors'] = pipeline_results.get('errors', [])
            
            # Determine status
            if result['errors']:
                result['status'] = 'error'
            elif result['eegnet_results'] or result['mirepnet_results']:
                result['status'] = 'success'
            else:
                result['status'] = 'no_results'
            
            # Print summary
            self._print_file_summary(result)
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            print(f"   ❌ Error processing {edf_path.name}: {e}")
        
        return result
    
    def _print_file_summary(self, result: Dict) -> None:
        """Print a summary of processing results for a single file."""
        print(f"\n📊 Results for {result['filename']}:")
        
        if result['eegnet_results']:
            summary = result['eegnet_results']['summary']
            print(f"   🧠 EEGNet (Startle Blink Detection):")
            print(f"      - Blinks detected: {summary['n_blinks']}")
            print(f"      - No blinks: {summary['n_no_blinks']}")
            print(f"      - Blink rate: {summary['blink_rate']:.2%}")
            print(f"      - Avg probability: {summary['avg_probability']:.3f}")
        
        if result['mirepnet_results']:
            mi = result['mirepnet_results']
            print(f"   🧠 MIRepNet (Motor Imagery Classification):")
            print(f"      - Predicted class: {mi['label']}")
            print(f"      - Confidence: {mi['confidence']:.2%}")
            print(f"      - Class probabilities:")
            for class_name, prob in mi['class_probabilities'].items():
                print(f"        • {class_name}: {prob:.2%}")
        
        if result['event_info']:
            print(f"   📋 Event file: Found ({result['event_info']['num_lines']} lines)")
        
        if result['errors']:
            print(f"   ⚠️  Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"      - {error}")
        
        print(f"   ✅ Status: {result['status']}")
    
    def process_folder(self, 
                      root_folder: str,
                      output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Process all EDF files in a folder structure.
        
        Parameters:
        -----------
        root_folder : str
            Root folder containing subject folders
        output_csv : Optional[str]
            Path to output CSV file for results
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all processing results
        """
        # Find all EDF files
        edf_files = self.find_edf_files(root_folder)
        
        if not edf_files:
            print("⚠️  No EDF files found!")
            return pd.DataFrame()
        
        print(f"\n🔄 Processing {len(edf_files)} file(s)...")
        
        # Process each file
        for i, file_info in enumerate(edf_files, 1):
            print(f"\n[{i}/{len(edf_files)}] Processing {file_info['filename']}...")
            
            result = self.process_single_file(
                edf_path=file_info['edf_path'],
                event_path=file_info['event_path']
            )
            
            self.results.append(result)
        
        # Convert to DataFrame
        df = self._results_to_dataframe()
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Results saved to: {output_csv}")
        
        # Print overall summary
        self._print_overall_summary(df)
        
        return df
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results list to a pandas DataFrame."""
        rows = []
        
        for result in self.results:
            row = {
                'file_path': result['file_path'],
                'filename': result['filename'],
                'subject': result['subject'],
                'timestamp': result['timestamp'],
                'status': result['status'],
                'num_errors': len(result['errors']),
                'errors': '; '.join(result['errors']) if result['errors'] else '',
            }
            
            # EEGNet results
            if result['eegnet_results']:
                summary = result['eegnet_results']['summary']
                row['eegnet_n_blinks'] = summary['n_blinks']
                row['eegnet_n_no_blinks'] = summary['n_no_blinks']
                row['eegnet_blink_rate'] = summary['blink_rate']
                row['eegnet_avg_probability'] = summary['avg_probability']
            else:
                row['eegnet_n_blinks'] = None
                row['eegnet_n_no_blinks'] = None
                row['eegnet_blink_rate'] = None
                row['eegnet_avg_probability'] = None
            
            # MIRepNet results
            if result['mirepnet_results']:
                mi = result['mirepnet_results']
                row['mirepnet_label'] = mi['label']
                row['mirepnet_class_index'] = mi['class_index']
                row['mirepnet_confidence'] = mi['confidence']
                # Add individual class probabilities
                for class_name, prob in mi['class_probabilities'].items():
                    row[f'mirepnet_prob_{class_name.lower()}'] = prob
            else:
                row['mirepnet_label'] = None
                row['mirepnet_class_index'] = None
                row['mirepnet_confidence'] = None
                for class_name in ['Forward', 'Left', 'Right', 'Stop']:
                    row[f'mirepnet_prob_{class_name.lower()}'] = None
            
            # Event file info
            row['has_event_file'] = result['event_info'] is not None
            row['event_file_lines'] = result['event_info']['num_lines'] if result['event_info'] else None
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _print_overall_summary(self, df: pd.DataFrame) -> None:
        """Print overall summary statistics."""
        print(f"\n{'='*70}")
        print("📊 OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_files = len(df)
        successful = len(df[df['status'] == 'success'])
        errors = len(df[df['status'] == 'error'])
        
        print(f"Total files processed: {total_files}")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Errors: {errors}")
        
        if successful > 0:
            # EEGNet summary
            if 'eegnet_n_blinks' in df.columns:
                total_blinks = df['eegnet_n_blinks'].sum()
                avg_blink_rate = df['eegnet_blink_rate'].mean()
                print(f"\n🧠 EEGNet Summary:")
                print(f"  - Total blinks detected: {int(total_blinks)}")
                print(f"  - Average blink rate: {avg_blink_rate:.2%}")
            
            # MIRepNet summary
            if 'mirepnet_label' in df.columns:
                print(f"\n🧠 MIRepNet Summary:")
                label_counts = df['mirepnet_label'].value_counts()
                for label, count in label_counts.items():
                    if pd.notna(label):
                        print(f"  - {label}: {count} ({count/successful*100:.1f}%)")
        
        print(f"{'='*70}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run batch processing."""
    
    # ========================================================================
    # CONFIGURATION - Update these paths for your setup
    # ========================================================================
    
    # Model paths
    EEGNET_MODEL_PATH = "D:/DEPI/finel depi/startle_blink_EEGNet_99_attention.keras"
    MIREPNET_MODEL_PATH = "D:/DEPI/finel depi/mirapnet_final_model.pth"
    
    # Data folder (update this to your actual data path)
    DATA_FOLDER = r"D:/DEPI/Finel/data/files/S007"
    
    # Output CSV file
    OUTPUT_CSV = "batch_processing_results.csv"
    
    # Device for PyTorch (use 'cuda' if GPU available)
    DEVICE = 'cpu'  # or 'cuda'
    
    # ========================================================================
    # Process files
    # ========================================================================
    
    print("=" * 70)
    print("🚀 EEG Batch Processing Pipeline")
    print("=" * 70)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("=" * 70)
    
    # Initialize processor
    processor = EDFBatchProcessor(
        eegnet_model_path=EEGNET_MODEL_PATH,
        mirepnet_model_path=MIREPNET_MODEL_PATH,
        device=DEVICE
    )
    
    # Process all files
    results_df = processor.process_folder(
        root_folder=DATA_FOLDER,
        output_csv=OUTPUT_CSV
    )
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Results DataFrame shape: {results_df.shape}")
    
    return results_df


if __name__ == "__main__":
    # Run batch processing
    results = main()
    
    # Optionally, display results
    if not results.empty:
        print("\n📋 Sample of results:")
        print(results.head())

