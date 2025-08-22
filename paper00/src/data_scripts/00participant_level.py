from shared_utils import *
import numpy as np

# Configuration constants
PARTICIPANT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TRAIN_PERCENT = 0.6
DEV_PERCENT = 0.2
TEST_PERCENT = 1 - TRAIN_PERCENT - DEV_PERCENT
SAVE_DIR = '/home/kuba/projects/puff/paper00/experiments/00-par-split/data'

# Create configuration
config = create_base_config(
    target_labels=['puff', 'puffs'],
    window_size=1024,
    step_size=1024,
    use_gyro=True,
    random_seed=70,
    percent_negative_windows=0.5,
    threshold_gap_minutes=5,
    label_value=1
)

def main():
    print("Starting Participant-Level Data Preparation")
    print(f"Participants: {PARTICIPANT_IDS}")
    print(f"Split ratios - Train: {TRAIN_PERCENT}, Dev: {DEV_PERCENT}, Test: {TEST_PERCENT}")
    print()

    # Validate configuration
    validate_config(config)
    validate_splits(TRAIN_PERCENT, DEV_PERCENT, TEST_PERCENT)

    # Create participant splits
    np.random.seed(config['random_seed'])
    random_perm = np.random.permutation(len(PARTICIPANT_IDS))
    train_size = int(len(random_perm) * TRAIN_PERCENT)
    dev_size = int(len(random_perm) * DEV_PERCENT)

    train_ids = [PARTICIPANT_IDS[i] for i in random_perm[:train_size]]
    dev_ids = [PARTICIPANT_IDS[i] for i in random_perm[train_size:train_size + dev_size]]
    test_ids = [PARTICIPANT_IDS[i] for i in random_perm[train_size + dev_size:]]

    print(f'TRAIN ids: {train_ids}')
    print(f'DEV ids: {dev_ids}')
    print(f'TEST ids: {test_ids}')
    print()

    # Create datasets
    print("Creating training dataset...")
    train_X, train_y = make_dataset_from_participants(train_ids, config)
    save_dataset(train_X, train_y, SAVE_DIR, "train")

    print("Creating development dataset...")
    dev_X, dev_y = make_dataset_from_participants(dev_ids, config)
    save_dataset(dev_X, dev_y, SAVE_DIR, "dev")

    print("Creating test dataset...")
    test_X, test_y = make_dataset_from_participants(test_ids, config)
    save_dataset(test_X, test_y, SAVE_DIR, "test")

    # Save configuration
    config_to_save = {
        'splits': {
            'train_ids': train_ids, 
            'dev_ids': dev_ids, 
            'test_ids': test_ids
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
    
    print("Participant-level data preparation completed!")

if __name__ == "__main__":
    main()