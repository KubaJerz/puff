#!/usr/bin/env python3
"""
K-Fold Customizer Script

Processes existing K-fold cross-validation results, creates new fine-tuned datasets,
and runs new experiments using the ExperimentBuilder framework.
"""

import os
import sys
import argparse
import toml
import logging
from pathlib import Path
from typing import List, Optional
import importlib.util

# Required imports
try:
    spec = importlib.util.spec_from_file_location('03test_finetune', '/home/kuba/projects/puff/paper00/src/data_scripts/03test_finetune.py')
    finetune = importlib.util.module_from_spec(spec)
    sys.modules['finetume'] = finetune
    spec.loader.exec_module(finetune)
    finetune_main = finetune.main
    from experiment_builder import ExperimentBuilder
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure that path to 03test_finetune.py datascript is correct in this script and experiment_builder are in your Python path")
    sys.exit(1)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="K-Fold Customizer: Process existing K-fold results and create new fine-tuned experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python k_fold_customizer.py config.toml
        """
    )
    
    # Single required argument: path to config TOML file
    parser.add_argument(
        'config_toml',
        type=str,
        help='Path to configuration TOML file'
    )
    
    return parser.parse_args()


def load_and_validate_config(config_path: str, logger):
    """Load and validate the configuration TOML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)
    
    # Validate required sections and fields
    required_fields = {
        'input_kfold_dir': str,
        'name': str,
        'expt_dir': str,
        'percent_old_train': float,
        'test_split_ratio': float,
        'use_optimizer_state': bool
    }
    
    required_sections = ['meta_data', 'static']
    
    errors = []
    
    # Check required top-level fields
    for field, expected_type in required_fields.items():
        if field not in config:
            errors.append(f"Missing required field: {field}")
            continue
        
        if field in ['percent_old_train', 'test_split_ratio']:
            if not isinstance(config[field], (int, float)):
                errors.append(f"Field '{field}' must be a number")
            elif not 0.0 <= config[field] <= 1.0:
                errors.append(f"Field '{field}' must be between 0.0 and 1.0, got: {config[field]}")
        elif field == 'use_optimizer_state':
            if not isinstance(config[field], bool):
                errors.append(f"Field '{field}' must be a boolean")
        else:
            if not isinstance(config[field], str):
                errors.append(f"Field '{field}' must be a string")
    
    # Check required sections exist
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: [{section}]")
    
    # Validate paths exist
    if 'input_kfold_dir' in config:
        input_path = Path(config['input_kfold_dir'])
        if not input_path.exists():
            errors.append(f"Input K-fold directory does not exist: {config['input_kfold_dir']}")
        elif not input_path.is_dir():
            errors.append(f"Input path is not a directory: {config['input_kfold_dir']}")
    
    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)
    
    logger.info("Configuration validation passed")
    return config

def save_config(config, save_dir, logger):
    # Save TOML configuration
    config_path = save_dir / "config.toml"
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    
    logger.info(f"Created configuration file: {config_path}")


def validate_arguments(args, logger):
    """Validate command line arguments.""" 
    # This function is now simplified since we only have one argument
    config_path = Path(args.config_toml)
    if not config_path.exists():
        logger.error(f"Configuration file does not exist: {args.config_toml}")
        sys.exit(1)
    
    logger.info("Arguments validation passed")


def detect_fold_directories(input_dir: Path, logger) -> List[str]:
    """Detect all fold-* directories in the input path."""
    fold_dirs = []
    
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith('fold-'):
            fold_dirs.append(item.name)
    
    fold_dirs.sort()  # Ensure consistent ordering
    
    if not fold_dirs:
        logger.error(f"No fold directories found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Detected {len(fold_dirs)} fold directories: {fold_dirs}")
    return fold_dirs


def create_output_structure(input_dir: Path, config: dict, fold_dirs: List[str], logger):
    """Create the output directory structure."""
    # Use expt_dir from config as the parent directory
    output_dir = Path(config['expt_dir']) / config['name']
    
    if output_dir.exists():
        logger.warning(f"Output directory already exists: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create fold subdirectories
    for fold_name in fold_dirs:
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(exist_ok=True)
        
        # Create data subdirectory
        # data_dir = fold_dir / "data"
        # data_dir.mkdir(exist_ok=True)

        runs_dir = fold_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created fold directory structure: {fold_dir}")
    
    return output_dir


def process_fold_data(input_fold_dir: Path, output_fold_dir: Path, 
                     percent_old_train: float, test_split_ratio: float, logger) -> bool:
    """Process data for a single fold using 03test_finetune.py."""
    try:
        old_data_path = str(input_fold_dir / "data")
        new_experiment_path = output_fold_dir
        
        # Check if input data directory exists
        if not Path(old_data_path).exists():
            logger.error(f"Input data directory does not exist: {old_data_path}")
            return False
        
        logger.info(f"Processing fold data: {old_data_path} -> {new_experiment_path}/data")
        
        # Call the finetune main function
        finetune_main(old_data_path, new_experiment_path, percent_old_train, test_split_ratio)
        
        logger.info(f"Successfully processed fold data for {output_fold_dir.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process fold data for {output_fold_dir.name}: {e}")
        return False


def create_fold_config(config: dict, output_fold_dir: Path, 
                      input_fold_dir: Path, logger) -> bool:
    """Create customized TOML configuration for a fold."""
    try:
        # Create a deep copy of the template sections from the main config
        import copy
        fold_config = {}
        
        # Copy all sections except the customizer-specific fields
        customizer_fields = {'input_kfold_dir', 'name', 'expt_dir', 'percent_old_train', 
                            'test_split_ratio', 'use_optimizer_state'}
        
        for key, value in config.items():
            if key not in customizer_fields:
                fold_config[key] = copy.deepcopy(value)
        
        fold_name = output_fold_dir.name
        
        # Modify specific fields for this fold
        if 'meta_data' not in fold_config:
            fold_config['meta_data'] = {}
        fold_config['meta_data']['name'] = 'customize'
        fold_config['meta_data']['expt_dir'] = str(output_fold_dir / "runs")
        
        # Set data paths (replace None values or update existing paths)
        if 'static' not in fold_config:
            fold_config['static'] = {}
        if 'data' not in fold_config['static']:
            fold_config['static']['data'] = {}
        
        data_path = str(output_fold_dir / "data")
        fold_config['static']['data']['train_path'] = data_path
        fold_config['static']['data']['dev_path'] = data_path
        fold_config['static']['data']['test_path'] = data_path
        
        # Add model checkpoint path if exists
        model_checkpoint = input_fold_dir / "runs" / "k-fold" / "0" / "best_dev_model.pt"
        if model_checkpoint.exists():
            if 'model' not in fold_config['static']:
                fold_config['static']['model'] = {}
            fold_config['static']['model']['model_weights'] = str(model_checkpoint)
            logger.info(f"Loaded model weights: {model_checkpoint}")
        else:
            logger.warning(f"Model checkpoint not found: {model_checkpoint}")
            # Remove model_weights if it was set to None in the template
            if 'static' in fold_config and 'model' in fold_config['static']:
                fold_config['static']['model'].pop('model_weights', None)
        
        # Add optimizer checkpoint path if flag is set
        if config['use_optimizer_state']:
            optimizer_checkpoint = input_fold_dir / "runs" / "k-fold" / "0" / "best_dev_optim.pt"
            if optimizer_checkpoint.exists():
                if 'optimizer' not in fold_config['static']:
                    fold_config['static']['optimizer'] = {}
                fold_config['static']['optimizer']['optimizer_weights'] = str(optimizer_checkpoint)
                logger.info(f"Loaded optimizer state: {optimizer_checkpoint}")
            else:
                logger.warning(f"Optimizer checkpoint not found: {optimizer_checkpoint}")
                # Remove optimizer_weights if it was set to None in the template
                if 'static' in fold_config and 'optimizer' in fold_config['static']:
                    fold_config['static']['optimizer'].pop('optimizer_weights', None)
        else:
            # Remove optimizer_weights if not using optimizer state
            if 'static' in fold_config and 'optimizer' in fold_config['static']:
                fold_config['static']['optimizer'].pop('optimizer_weights', None)
        
        # Save TOML configuration

        save_config(fold_config, output_fold_dir, logger)
        return True
        
    except Exception as e:
        logger.error(f"Failed to create configuration for {output_fold_dir.name}: {e}")
        return False


def run_experiment(output_fold_dir: Path, logger) -> bool:
    """Run experiment using ExperimentBuilder."""
    try:
        config_path = output_fold_dir / "config.toml"
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        logger.info(f"Starting experiment for {output_fold_dir.name}")
        
        # Create and run experiment
        builder = ExperimentBuilder(str(config_path))
        builder.build_experiment_runs()
        builder.start_runs()
        
        logger.info(f"Successfully completed experiment for {output_fold_dir.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to run experiment for {output_fold_dir.name}: {e}")
        return False


def main():
    """Main function to orchestrate the K-fold customization process."""
    logger = setup_logging()
    
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args, logger)
    
    # Load and validate configuration
    config = load_and_validate_config(args.config_toml, logger)
    
    # Convert paths to Path objects
    input_dir = Path(config['input_kfold_dir'])
    
    logger.info("Starting K-Fold Customizer")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output name: {config['name']}")
    logger.info(f"Experiments directory: {config['expt_dir']}")
    logger.info(f"Percent old train: {config['percent_old_train']}")
    logger.info(f"Test split ratio: {config['test_split_ratio']}")
    logger.info(f"Use optimizer state: {config['use_optimizer_state']}")
    
    # Step 1: Detect fold directories and create output structure
    fold_dirs = detect_fold_directories(input_dir, logger)
    output_dir = create_output_structure(input_dir, config, fold_dirs, logger)
    save_config(config, output_dir, logger)
    
    # Process each fold
    successful_folds = 0
    failed_folds = 0
    
    for fold_name in fold_dirs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {fold_name}")
        logger.info(f"{'='*50}")
        
        input_fold_dir = input_dir / fold_name
        output_fold_dir = output_dir / fold_name
        
        fold_success = True
        
        # Step 2: Process fold data
        if not process_fold_data(input_fold_dir, output_fold_dir, 
                                config['percent_old_train'], config['test_split_ratio'], logger):
            fold_success = False
        
        # Step 3: Create TOML configuration
        if fold_success and not create_fold_config(config, output_fold_dir, 
                                                  input_fold_dir, logger):
            fold_success = False
        
        # Step 4: Run experiment
        if fold_success and not run_experiment(output_fold_dir, logger):
            fold_success = False
        
        if fold_success:
            successful_folds += 1
            logger.info(f"✓ Successfully completed {fold_name}")
        else:
            failed_folds += 1
            logger.error(f"✗ Failed to process {fold_name}")
    
    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info("K-Fold Customizer Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total folds processed: {len(fold_dirs)}")
    logger.info(f"Successful folds: {successful_folds}")
    logger.info(f"Failed folds: {failed_folds}")
    logger.info(f"Output directory: {output_dir}")
    
    if failed_folds > 0:
        logger.warning(f"Some folds failed to process. Check logs above for details.")
        sys.exit(1)
    else:
        logger.info("All folds processed successfully!")


if __name__ == "__main__":
    main()