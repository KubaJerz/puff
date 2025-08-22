from shared_utils import *
import numpy as np
import os

# K-FOLD SPECIFIC CONSTANTS
PARTICIPANT_IDS = [1, 3, 6, 7, 10]  # List of participant IDs to include
NUM_FOLDS = 5  # Number of folds for cross-validation
KFOLD_BASE_DIR = '/home/kuba/Desktop/k-fold'  # Base directory containing fold subdirectories

# Configuration constants
TRAIN_PERCENT = 0.6  # Percentage for training set (applied to non-test participants)
DEV_PERCENT = 1 - TRAIN_PERCENT

# Create configuration
config = create_base_config(
    target_labels=['puff', 'puffs'],
    window_size=256,
    step_size=256,
    use_gyro=False,
    random_seed=70,
    percent_negative_windows=0.5,
    threshold_gap_minutes=30,
    label_value=1,
    resample=False
)

def validate_kfold_structure():
    """Check that k-fold directory structure is correct."""
    if not os.path.exists(KFOLD_BASE_DIR):
        raise FileNotFoundError(f"K-fold base directory does not exist: {KFOLD_BASE_DIR}")
    
    # Check for exactly NUM_FOLDS subdirectories
    expected_dirs = [f"fold-{i}" for i in range(NUM_FOLDS)]
    existing_dirs = [d for d in os.listdir(KFOLD_BASE_DIR) 
                    if os.path.isdir(os.path.join(KFOLD_BASE_DIR, d))]
    
    missing_dirs = set(expected_dirs) - set(existing_dirs)
    extra_dirs = set(existing_dirs) - set(expected_dirs)
    
    if missing_dirs:
        raise FileNotFoundError(f"Missing fold directories: {sorted(missing_dirs)}")
    
    if extra_dirs:
        print(f"Warning: Found unexpected directories: {sorted(extra_dirs)}")
    
    print(f"Validated k-fold structure with {NUM_FOLDS} folds")

def validate_kfold_parameters():
    """Validate k-fold parameters."""
    if not PARTICIPANT_IDS:
        raise ValueError("PARTICIPANT_IDS cannot be empty")
    
    if TRAIN_PERCENT + DEV_PERCENT != 1.0:
        raise ValueError(f"TRAIN_PERCENT + DEV_PERCENT must be == 1.0, got {TRAIN_PERCENT + DEV_PERCENT}")
    
    if NUM_FOLDS > len(PARTICIPANT_IDS):
        print(f"Warning: NUM_FOLDS ({NUM_FOLDS}) > participants ({len(PARTICIPANT_IDS)}), "
              f"this will create leave-one-out style validation")

def generate_kfold_splits():
    """Generate all k-fold participant splits."""
    participants_per_fold = len(PARTICIPANT_IDS) // NUM_FOLDS
    remainder = len(PARTICIPANT_IDS) % NUM_FOLDS
    
    splits = []
    start_idx = 0
    
    for fold_k in range(NUM_FOLDS):
        # Calculate test set size for this fold
        if fold_k < remainder:
            test_size = participants_per_fold + 1
        else:
            test_size = participants_per_fold
        
        # Get test participants for this fold
        test_ids = PARTICIPANT_IDS[start_idx:start_idx + test_size]
        
        # Get remaining participants for train/dev split
        remaining_ids = [p for p in PARTICIPANT_IDS if p not in test_ids]
        
        # Split remaining participants into train/dev
        np.random.seed(config['random_seed'])
        random_perm = np.random.permutation(remaining_ids)
        train_size = int(len(random_perm) * TRAIN_PERCENT)
        
        train_ids = random_perm[:train_size].tolist()
        dev_ids = random_perm[train_size:].tolist()
        
        splits.append({
            'fold': fold_k,
            'train_ids': train_ids,
            'dev_ids': dev_ids,
            'test_ids': test_ids
        })
        
        start_idx += test_size
    
    return splits

def save_fold_config(fold_num, train_ids, dev_ids, test_ids, save_dir):
    """Save fold-specific configuration."""
    config_dict = {
        "kfold": {
            "fold_number": fold_num,
            "total_folds": NUM_FOLDS,
            "participant_ids": PARTICIPANT_IDS
        },
        "splits": {
            "train_ids": train_ids,
            "dev_ids": dev_ids,
            "test_ids": test_ids
        },
        "paths": {
            "kfold_base_dir": KFOLD_BASE_DIR,
            "save_dir": save_dir
        },
        "experiment": config,
        "split_percentages": {
            "train_percent": TRAIN_PERCENT,
            "dev_percent": DEV_PERCENT,
            "test_percent": 1.0 - TRAIN_PERCENT - DEV_PERCENT,
            "num_participants": len(PARTICIPANT_IDS)
        }
    }
    
    save_config(config_dict, save_dir)

def process_fold(fold_num, train_ids, dev_ids, test_ids):
    """Process a single fold."""
    print(f"\n\nProcessing fold {fold_num}/{NUM_FOLDS}...")
    print(f"TRAIN ids: {train_ids}")
    print(f"DEV ids: {dev_ids}")
    print(f"TEST ids: {test_ids}")
    
    # Set up fold-specific save directory
    fold_save_dir = os.path.join(KFOLD_BASE_DIR, f"fold-{fold_num}", "data")
    os.makedirs(fold_save_dir, exist_ok=True)
    
    # Save fold configuration
    save_fold_config(fold_num, train_ids, dev_ids, test_ids, fold_save_dir)
    
    # Create and save datasets
    print("Creating training dataset...")
    train_X, train_y = make_dataset_from_participants(train_ids, config)
    save_dataset(train_X, train_y, fold_save_dir, "train")
    
    print("Creating development dataset...")
    dev_X, dev_y = make_dataset_from_participants(dev_ids, config)
    save_dataset(dev_X, dev_y, fold_save_dir, "dev")
    
    print("Creating test dataset...")
    test_X, test_y = make_dataset_from_participants(test_ids, config)
    save_dataset(test_X, test_y, fold_save_dir, "test")

def main():
    """Main execution function."""
    print("Starting K-Fold Cross-Validation Data Preparation")
    print(f"Participants: {PARTICIPANT_IDS}")
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Base directory: {KFOLD_BASE_DIR}")
    print()

    # Validate configuration and directory structure
    validate_config(config)
    validate_kfold_parameters()
    validate_kfold_structure()
    validate_splits(TRAIN_PERCENT, DEV_PERCENT, 0)

    # Generate k-fold splits
    print("Generating k-fold splits...")
    fold_splits = generate_kfold_splits()

    # Process each fold
    for split in fold_splits:
        process_fold(
            split['fold'],
            split['train_ids'],
            split['dev_ids'],
            split['test_ids']
        )

    print("\nK-Fold cross-validation data preparation completed!")

if __name__ == "__main__":
    main()