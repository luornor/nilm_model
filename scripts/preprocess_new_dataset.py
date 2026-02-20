"""
Template for preprocessing new datasets into NILM framework format.

This script converts external datasets to the standardized CSV format:
    file, t_sec, P, y_<appliance1>, y_<appliance2>, ...

Usage:
    python scripts/preprocess_new_dataset.py \
        --dataset-path Dataset/NewDataset \
        --dataset-name dataset1 \
        --output Exports/dataset1_1s.csv \
        --sampling-rate 1.0

Author: NILM Framework
Date: 2026-02-13
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def load_taxonomy():
    """Load device taxonomy for name normalization."""
    taxonomy_path = Path('configs/device_taxonomy.yaml')
    if taxonomy_path.exists():
        with open(taxonomy_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def normalize_device_name(device_name: str, taxonomy: Dict) -> str:
    """
    Normalize device names using taxonomy mappings.
    
    Args:
        device_name: Original device name
        taxonomy: Device taxonomy dict
    
    Returns:
        Normalized device name
    """
    if 'device_mappings' in taxonomy:
        return taxonomy['device_mappings'].get(device_name, device_name)
    return device_name


def parse_plaid_format(data_path: Path) -> pd.DataFrame:
    """
    Parse PLAID dataset format.
    
    PLAID structure:
    - metadata.json: appliance info
    - submetered/*.mat: individual appliance measurements
    - aggregated/*.mat: combined power measurements
    
    Returns:
        DataFrame with columns: file, t_sec, P, y_*
    """
    raise NotImplementedError("Implement based on dataset format")


def parse_blued_format(data_path: Path) -> pd.DataFrame:
    """
    Parse BLUED dataset format.
    
    BLUED structure:
    - Phase A/B voltage and current at 12kHz
    - Event labels for appliances
    
    Returns:
        DataFrame with columns: file, t_sec, P, y_*
    """
    raise NotImplementedError("Implement based on dataset format")


def parse_ukdale_format(data_path: Path) -> pd.DataFrame:
    """
    Parse UK-DALE dataset format.
    
    UK-DALE structure:
    - HDF5 files with aggregate and sub-metered data
    - Metadata in YAML
    
    Returns:
        DataFrame with columns: file, t_sec, P, y_*
    """
    raise NotImplementedError("Implement based on dataset format")


def parse_custom_format(data_path: Path, format_type: str) -> pd.DataFrame:
    """
    Parse custom dataset format.
    
    TODO: Implement based on your specific dataset structure
    
    Args:
        data_path: Path to dataset directory
        format_type: Type of format (csv, hdf5, mat, etc.)
    
    Returns:
        DataFrame with columns: file, t_sec, P, y_*
    """
    if format_type == 'csv':
        # Example: CSV with columns [timestamp, power, appliance_states]
        df = pd.read_csv(data_path)
        # TODO: Rename columns, create binary labels, etc.
        
    elif format_type == 'mat':
        # Example: MATLAB files with power and event data
        from scipy.io import loadmat
        data = loadmat(data_path)
        # TODO: Extract power, events, create sliding windows
        
    elif format_type == 'hdf5':
        # Example: HDF5 with hierarchical structure
        import h5py
        with h5py.File(data_path, 'r') as f:
            # TODO: Extract relevant data
            pass
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return pd.DataFrame()


def compute_power_from_vi(voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous power from voltage and current.
    
    Args:
        voltage: Voltage waveform (V)
        current: Current waveform (A)
    
    Returns:
        Instantaneous power (W)
    """
    return voltage * current


def bin_data_to_seconds(df: pd.DataFrame, bin_size: float = 1.0) -> pd.DataFrame:
    """
    Aggregate high-frequency data into fixed time bins.
    
    Args:
        df: DataFrame with high-resolution time series
        bin_size: Bin size in seconds
    
    Returns:
        Binned DataFrame
    """
    df['time_bin'] = (df['t_sec'] // bin_size) * bin_size
    
    # Aggregate power (mean) and appliance states (mode/max)
    agg_dict = {'P': 'mean'}
    
    for col in df.columns:
        if col.startswith('y_'):
            agg_dict[col] = lambda x: (x.mean() > 0.5).astype(int)  # Binary threshold
    
    binned = df.groupby(['file', 'time_bin']).agg(agg_dict).reset_index()
    binned.rename(columns={'time_bin': 't_sec'}, inplace=True)
    
    return binned


def extract_appliance_states_from_events(events: np.ndarray, 
                                          timestamps: np.ndarray,
                                          threshold: float = 0.5) -> np.ndarray:
    """
    Convert event markers to binary ON/OFF states.
    
    Args:
        events: Event markers (+1 for ON, -1 for OFF, 0 for no change)
        timestamps: Time points for each sample
        threshold: Threshold for binarization
    
    Returns:
        Binary state array (0 or 1)
    """
    cumulative_state = np.cumsum(events)
    binary_state = (cumulative_state > threshold).astype(int)
    return binary_state


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame meets framework requirements.
    
    Required columns:
    - file: str
    - t_sec: float
    - P: float
    - y_*: int (0 or 1)
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns
    if 'file' not in df.columns:
        errors.append("Missing 'file' column")
    if 't_sec' not in df.columns:
        errors.append("Missing 't_sec' column")
    if 'P' not in df.columns:
        errors.append("Missing 'P' column")
    
    # Check for at least one appliance column
    appliance_cols = [col for col in df.columns if col.startswith('y_')]
    if len(appliance_cols) == 0:
        errors.append("No appliance columns (y_*) found")
    
    # Check data types and ranges
    if 'P' in df.columns:
        if df['P'].isnull().any():
            errors.append("'P' column contains NaN values")
        if (df['P'] < 0).any():
            errors.append("'P' column contains negative values")
    
    # Check binary states
    for col in appliance_cols:
        unique_vals = df[col].unique()
        if not set(unique_vals).issubset({0, 1}):
            errors.append(f"{col} is not binary (contains values other than 0/1)")
    
    return (len(errors) == 0, errors)


def generate_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate statistics about the dataset.
    
    Returns:
        Dictionary with dataset metadata
    """
    appliance_cols = [col for col in df.columns if col.startswith('y_')]
    
    stats = {
        'total_samples': len(df),
        'duration_hours': df['t_sec'].max() / 3600,
        'num_files': df['file'].nunique(),
        'num_appliances': len(appliance_cols),
        'appliances': {},
        'power_stats': {
            'mean': df['P'].mean(),
            'std': df['P'].std(),
            'min': df['P'].min(),
            'max': df['P'].max(),
        }
    }
    
    for col in appliance_cols:
        appliance_name = col[2:]  # Remove 'y_' prefix
        on_mask = df[col] == 1
        
        stats['appliances'][appliance_name] = {
            'on_rate': on_mask.mean(),
            'num_on_samples': on_mask.sum(),
            'num_events': df[col].diff().abs().sum() / 2,  # Transitions / 2
            'avg_power_on': df.loc[on_mask, 'P'].mean() if on_mask.any() else 0,
            'avg_power_off': df.loc[~on_mask, 'P'].mean() if (~on_mask).any() else 0,
            'power_delta': df.loc[on_mask, 'P'].mean() - df.loc[~on_mask, 'P'].mean() if on_mask.any() and (~on_mask).any() else 0,
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess new dataset for NILM framework')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Dataset name (e.g., dataset1, ukdale, plaid)')
    parser.add_argument('--format', type=str, default='custom',
                        choices=['custom', 'plaid', 'blued', 'ukdale', 'csv', 'mat', 'hdf5'],
                        help='Dataset format')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--sampling-rate', type=float, default=1.0,
                        help='Target sampling rate in seconds (default: 1.0)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks on output')
    parser.add_argument('--stats', action='store_true',
                        help='Generate and save statistics')
    
    args = parser.parse_args()
    
    # Load taxonomy for name normalization
    taxonomy = load_taxonomy()
    print(f"Loaded device taxonomy with {len(taxonomy.get('device_mappings', {}))} mappings")
    
    # Parse dataset based on format
    print(f"Parsing dataset: {args.dataset_name} (format: {args.format})")
    data_path = Path(args.dataset_path)
    
    if args.format == 'plaid':
        df = parse_plaid_format(data_path)
    elif args.format == 'blued':
        df = parse_blued_format(data_path)
    elif args.format == 'ukdale':
        df = parse_ukdale_format(data_path)
    else:
        df = parse_custom_format(data_path, args.format)
    
    # Normalize device names
    print("Normalizing device names...")
    for col in df.columns:
        if col.startswith('y_'):
            old_name = col[2:]
            new_name = normalize_device_name(old_name, taxonomy)
            if old_name != new_name:
                df.rename(columns={col: f'y_{new_name}'}, inplace=True)
                print(f"  Renamed: {old_name} -> {new_name}")
    
    # Validate output
    if args.validate:
        print("\nValidating output...")
        is_valid, errors = validate_dataframe(df)
        if not is_valid:
            print("Validation FAILED:")
            for error in errors:
                print(f"  - {error}")
            return
        print("Validation PASSED ✓")
    
    # Generate statistics
    if args.stats:
        print("\nGenerating statistics...")
        stats = generate_data_statistics(df)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Duration: {stats['duration_hours']:.2f} hours")
        print(f"  Number of files: {stats['num_files']}")
        print(f"  Number of appliances: {stats['num_appliances']}")
        print(f"  Power: mean={stats['power_stats']['mean']:.1f}W, "
              f"std={stats['power_stats']['std']:.1f}W, "
              f"max={stats['power_stats']['max']:.1f}W")
        
        print(f"\nPer-Appliance Statistics:")
        for appliance, app_stats in stats['appliances'].items():
            print(f"  {appliance}:")
            print(f"    ON rate: {app_stats['on_rate']*100:.1f}%")
            print(f"    ON samples: {app_stats['num_on_samples']:,}")
            print(f"    Events: {app_stats['num_events']:.0f}")
            print(f"    Power delta: {app_stats['power_delta']:.1f}W")
        
        # Save statistics to JSON
        stats_path = Path(args.output).parent / f'{args.dataset_name}_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} samples")
    
    print("\n✓ Preprocessing complete!")


if __name__ == '__main__':
    main()
