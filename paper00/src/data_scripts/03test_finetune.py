#!/usr/bin/env python3
"""
test_finetune.py

A script for processing old training data and creating new train/dev/test splits
for fine-tuning experiments. Can be run standalone or imported as a module.
"""

import os
import torch
import toml
from pathlib import Path
from typing import Tuple, Dict, Any


def main(old_data_path: str, new_experiment_path: str, percent_old_train: float, test_split_ratio: float = 1/3) -> None:
    """
    Process old training data and create new train/dev/test splits for fine-tuning experiments.
    
    Args:
        old_data_path (str): Path to directory containing train.pt and test.pt files
        new_experiment_path (str): Path to new experiment directory (will create data/ subdirectory)
        percent_old_train (float): Percentage of old training data to include (0.0 to 1.0)
        test_split_ratio (float): Ratio for 3-way split of old test data (default: 1/3)
    
    Raises:
        FileNotFoundError: If required files or directories don't exist
        ValueError: If parameters are out of valid range or data format is invalid
        RuntimeError: If tensor operations fail
    """
    
    # Step 1: Load and Validate Data
    print("Step 1: Loading and validating data...")
    old_train_data, old_test_data = _load_and_validate_data(old_data_path)
    
    # Step 2: Validate Parameters
    print("\nStep 2: Validating parameters...")
    _validate_parameters(percent_old_train, test_split_ratio, old_test_data)
    
    # Step 3: Split Old Test Data
    print("\nStep 3: Splitting old test data...")
    new_train_addition, new_dev, new_test = _split_test_data(old_test_data, test_split_ratio)
    
    # Step 4: Create New Training Set
    print("\nStep 4: Creating new training set...")
    new_train = _create_new_training_set(old_train_data, new_train_addition, percent_old_train)
    
    # Step 5: Create Output Directory and Save Datasets
    print("\nStep 5: Saving new datasets...")
    data_dir = _create_output_directory(new_experiment_path)
    _save_datasets(data_dir, new_train, new_dev, new_test)
    
    # Step 6: Save Configuration
    print("\n\nStep 6: Saving configuration...")
    config_info = _create_config_info(
        old_data_path, new_experiment_path, percent_old_train, test_split_ratio,
        old_train_data, old_test_data, new_train, new_dev, new_test
    )
    _save_config(data_dir, config_info)
    
    # Print success summary
    _print_success_summary(config_info)


def _load_and_validate_data(old_data_path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Load and validate the old training and test data."""
    old_path = Path(old_data_path)
    
    # Check if directory exists
    if not old_path.exists() or not old_path.is_dir():
        raise FileNotFoundError(f"Directory {old_data_path} does not exist")
    
    train_file = old_path / "train.pt"
    test_file = old_path / "test.pt"
    
    # Check if required files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Required file {train_file} not found")
    if not test_file.exists():
        raise FileNotFoundError(f"Required file {test_file} not found")
    
    # Load data files
    try:
        train_data = torch.load(train_file, weights_only=True)
        test_data = torch.load(test_file, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load data files: {e}")
    
    # Validate data format
    def validate_data_format(data, filepath):
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(f"Invalid data format in {filepath}: expected tuple (X, y) of PyTorch tensors")
        
        X, y = data
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError(f"Invalid data format in {filepath}: expected tuple (X, y) of PyTorch tensors")
        
        return X, y
    
    train_X, train_y = validate_data_format(train_data, train_file)
    test_X, test_y = validate_data_format(test_data, test_file)
    
    # Validate tensor shape compatibility
    if train_X.shape[1:] != test_X.shape[1:]:  # Compare all dimensions except batch size
        raise ValueError(f"Tensor shape mismatch: train X shape {train_X.shape} incompatible with test X shape {test_X.shape}")
    
    if train_y.shape[1:] != test_y.shape[1:]:  # Compare all dimensions except batch size
        raise ValueError(f"Tensor shape mismatch: train y shape {train_y.shape} incompatible with test y shape {test_y.shape}")
    
    print(f"✓ Loaded train data: X{train_X.shape}, y{train_y.shape}")
    print(f"✓ Loaded test data: X{test_X.shape}, y{test_y.shape}")
    
    return (train_X, train_y), (test_X, test_y)


def _validate_parameters(percent_old_train: float, test_split_ratio: float, old_test_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Validate input parameters."""
    if not 0.0 <= percent_old_train <= 1.0:
        raise ValueError(f"Parameter percent_old_train must be between 0.0 and 1.0, got {percent_old_train}")
    
    if not 0.0 <= test_split_ratio <= 1.0:
        raise ValueError(f"Parameter test_split_ratio must be between 0.0 and 1.0, got {test_split_ratio}")
    
    test_X, test_y = old_test_data
    test_samples = test_X.shape[0]
    min_samples = 3  # Need at least 3 samples for 3-way split
    
    if test_samples < min_samples:
        raise ValueError(f"Insufficient test data: need at least {min_samples} samples for splitting, got {test_samples}")
    
    print(f"✓ Parameters validated: percent_old_train={percent_old_train}, test_split_ratio={test_split_ratio}")


def _split_test_data(old_test_data: Tuple[torch.Tensor, torch.Tensor], test_split_ratio: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Split old test data into three parts: new_train_addition, new_dev, new_test."""
    test_X, test_y = old_test_data
    total_samples = test_X.shape[0]
    
    # Calculate split sizes
    split_size = int(total_samples * test_split_ratio)
    
    # Create indices for splitting
    indices = torch.randperm(total_samples)
    
    # Split indices
    try:
        idx1 = indices[:split_size]
        idx2 = indices[split_size:2*split_size]
        idx3 = indices[2*split_size:]
    except Exception as e:
        raise IndexError(f"ERROR: your 'test_split_ratio' is off: {e}")

    # Split data
    new_train_addition = (test_X[idx1], test_y[idx1])
    new_dev = (test_X[idx2], test_y[idx2])
    new_test = (test_X[idx3], test_y[idx3])
    
    print(f"✓ Split test data: {len(idx1)} for train addition, {len(idx2)} for dev, {len(idx3)} for test")
    
    return new_train_addition, new_dev, new_test


def _create_new_training_set(old_train_data: Tuple[torch.Tensor, torch.Tensor], 
                           new_train_addition: Tuple[torch.Tensor, torch.Tensor], 
                           percent_old_train: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create new training set by combining subset of old training data with new train addition."""
    old_train_X, old_train_y = old_train_data
    addition_X, addition_y = new_train_addition
    
    # Calculate how many samples to take from old training data
    old_train_samples = old_train_X.shape[0]
    samples_to_take = int(old_train_samples * percent_old_train)
    
    # Randomly select subset of old training data
    indices = torch.randperm(old_train_samples)[:samples_to_take]
    old_train_subset_X = old_train_X[indices]
    old_train_subset_y = old_train_y[indices]
    
    # Combine old subset with new addition
    combined_X = torch.cat([old_train_subset_X, addition_X])
    combined_y = torch.cat([old_train_subset_y, addition_y])
    
    # SHUFFLE THE FINAL COMBINED DATASET
    total_samples = combined_X.shape[0]
    shuffle_indices = torch.randperm(total_samples)
    new_train_X = combined_X[shuffle_indices]
    new_train_y = combined_y[shuffle_indices]
    
    print(f"✓ Created new training set: {samples_to_take} from old train + {addition_X.shape[0]} from test split = {new_train_X.shape[0]} total")
    
    return (new_train_X, new_train_y)

def _create_output_directory(new_experiment_path: str) -> Path:
    """Create output directory structure."""
    experiment_path = Path(new_experiment_path)
    data_dir = experiment_path / "data"
    
    # Check if parent directory exists, create if needed
    if not experiment_path.parent.exists():
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create experiment and data directories
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure: {data_dir}")
    
    return data_dir


def _save_datasets(data_dir: Path, new_train: Tuple[torch.Tensor, torch.Tensor], 
                  new_dev: Tuple[torch.Tensor, torch.Tensor], 
                  new_test: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Save the new datasets to files."""
    print(f"The shape of the new train X data is: {new_train[0].shape} new train y data is: {new_train[1].shape}")
    print(f"The shape of the new dev X data is: {new_dev[0].shape} new dev y data is: {new_dev[1].shape}")
    print(f"The shape of the new test X data is: {new_test[0].shape} new test y data is: {new_test[1].shape}")

    try:
        torch.save(new_train, data_dir / "train.pt")
        torch.save(new_dev, data_dir / "dev.pt")
        torch.save(new_test, data_dir / "test.pt")
        
        print(f"✓ Saved datasets to {data_dir}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save datasets: {e}")


def _create_config_info(old_data_path: str, new_experiment_path: str, 
                       percent_old_train: float, test_split_ratio: float,
                       old_train_data: Tuple[torch.Tensor, torch.Tensor],
                       old_test_data: Tuple[torch.Tensor, torch.Tensor],
                       new_train: Tuple[torch.Tensor, torch.Tensor],
                       new_dev: Tuple[torch.Tensor, torch.Tensor],
                       new_test: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
    """Create configuration information dictionary."""
    return {
        "experiment": {
            "old_data_path": str(Path(old_data_path).resolve()),
            "new_experiment_path": str(Path(new_experiment_path).resolve()),
            "percent_old_train": percent_old_train,
            "test_split_ratio": test_split_ratio
        },
        "data_info": {
            "old_train_samples": old_train_data[0].shape[0],
            "old_test_samples": old_test_data[0].shape[0],
            "new_train_samples": new_train[0].shape[0],
            "new_dev_samples": new_dev[0].shape[0],
            "new_test_samples": new_test[0].shape[0]
        }
    }


def _save_config(data_dir: Path, config_info: Dict[str, Any]) -> None:
    """Save configuration to TOML file."""
    config_file = data_dir / "config.toml"
    
    try:
        with open(config_file, 'w') as f:
            toml.dump(config_info, f)
        
        print(f"✓ Saved configuration to {config_file}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save configuration: {e}")


def _print_success_summary(config_info: Dict[str, Any]) -> None:
    """Print success summary."""
    experiment_info = config_info["experiment"]
    data_info = config_info["data_info"]
    
    old_train_used = int(data_info["old_train_samples"] * experiment_info["percent_old_train"])
    percent = experiment_info["percent_old_train"]
    
    print("\n" + "="*60)
    print("Dataset creation completed successfully!")
    print(f"- Old train samples used: {old_train_used} / {data_info['old_train_samples']} ({percent:.1%})")
    print(f"- New train samples: {data_info['new_train_samples']}")
    print(f"- New dev samples: {data_info['new_dev_samples']}")
    print(f"- New test samples: {data_info['new_test_samples']}")
    print(f"- Files saved to: {experiment_info['new_experiment_path']}/data/")
    print(f"- Configuration saved to: {experiment_info['new_experiment_path']}/data/config.toml")
    print("="*60)


if __name__ == "__main__":
    # Hardcoded parameters for standalone execution
    OLD_DATA_PATH = "/home/kuba/Desktop/k-fold/fold-0/data"
    NEW_EXPERIMENT_PATH = "/home/kuba/projects/puff/test/experiments/test_custom_finetune"
    PERCENT_OLD_TRAIN = 0.75
    TEST_SPLIT_RATIO = 1/3
    
    try:
        main(OLD_DATA_PATH, NEW_EXPERIMENT_PATH, PERCENT_OLD_TRAIN, TEST_SPLIT_RATIO)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)