from utils import *
from sklearn.model_selection import train_test_split
import numpy as np

# Configuration constants
TRAIN_PERCENT = 0.6
DEV_PERCENT = 0.2
TEST_PERCENT = 0.2  # No test set in this configuration
SAVE_DIR = '/home/kuba/projects/puff/paper00/experiments/01-sesh-split/data'

# Create configuration
config = create_base_config(
    target_labels={'puff', 'puffs'}, 
    window_size=1024,
    step_size=1024,
    use_gyro=True,
    random_seed=70,
    percent_negative_windows=0.8,
    threshold_gap_minutes=5,
    label_value=1
)

def main():
    print("Starting Session-Level Data Preparation")
    print(f"Split ratios - Train: {TRAIN_PERCENT}, Dev: {DEV_PERCENT}")
    print()

    # Validate configuration
    validate_config(config)
    validate_splits(TRAIN_PERCENT, DEV_PERCENT, TEST_PERCENT)

    # Get all sessions from database
    print("Loading sessions from database...")
    all_sessions = get_all_sessions_from_db(config['target_labels'])
    print(f"Found {len(all_sessions)} sessions with target labels")

    # Split sessions using sklearn
    np.random.seed(config['random_seed'])
    
    if TEST_PERCENT > 0:
        train_sessions, temp_sessions = train_test_split(
            all_sessions, 
            train_size=TRAIN_PERCENT, 
            random_state=config['random_seed']
        )
        
        dev_sessions, test_sessions = train_test_split(
            temp_sessions,
            train_size=DEV_PERCENT/(DEV_PERCENT + TEST_PERCENT),
            random_state=config['random_seed']
        )
    else:
        train_sessions, dev_sessions = train_test_split(
            all_sessions,
            train_size=TRAIN_PERCENT,
            random_state=config['random_seed']
        )
        test_sessions = []

    print(f'TRAIN sessions: {len(train_sessions)}')
    print(f'DEV sessions: {len(dev_sessions)}')
    print(f'TEST sessions: {len(test_sessions)}')
    print()

    # Create datasets
    print("Creating training dataset...")
    train_X, train_y = make_dataset_from_sessions(train_sessions, config)
    save_dataset(train_X, train_y, SAVE_DIR, "train")

    print("Creating development dataset...")
    dev_X, dev_y = make_dataset_from_sessions(dev_sessions, config)
    save_dataset(dev_X, dev_y, SAVE_DIR, "dev")

    if test_sessions:
        print("Creating test dataset...")
        test_X, test_y = make_dataset_from_sessions(test_sessions, config)
        save_dataset(test_X, test_y, SAVE_DIR, "test")

    # Save configuration with session details
    config_to_save = {
        'splits': {
            'train_sessions': [f"{s[1]}_{s[0]['session_name']}" for s in train_sessions],
            'dev_sessions': [f"{s[1]}_{s[0]['session_name']}" for s in dev_sessions],
            'test_sessions': [f"{s[1]}_{s[0]['session_name']}" for s in test_sessions]
        },
        'experiment': config,
        'split_percentages': {
            'train': TRAIN_PERCENT, 
            'dev': DEV_PERCENT, 
            'test': TEST_PERCENT
        },
        'paths': {
            'save_dir': SAVE_DIR
        }
    }
    save_config(config_to_save, SAVE_DIR)
    
    print("Session-level data preparation completed!")

if __name__ == "__main__":
    main()