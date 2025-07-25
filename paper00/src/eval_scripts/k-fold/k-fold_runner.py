#!/usr/bin/env python3
"""
K-Fold Runner Script

Processes multiple subdirectories for machine learning experiments,
modifying TOML configuration files and running experiments for each subdirectory.
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import toml
import warnings

# Required imports
sys.path.insert(1, '/home/kuba/projects/puff/paper00/src/train_scripts')
from experiment_builder import ExperimentBuilder
from expt_runner import Expt_Runner


def validate_main_directory(main_dir_path):
    """
    Validate that the main directory exists and contains a TOML file.
    
    Args:
        main_dir_path (str): Absolute path to the main directory
        
    Returns:
        str: Path to the main TOML file
        
    Raises:
        FileNotFoundError: If main directory or TOML file doesn't exist
    """
    if not os.path.exists(main_dir_path):
        raise FileNotFoundError(f"Main directory '{main_dir_path}' does not exist")
    
    # Look for TOML files in the main directory
    toml_files = glob.glob(os.path.join(main_dir_path, "*.toml"))
    
    if not toml_files:
        raise FileNotFoundError(f"No TOML configuration file found in main directory '{main_dir_path}'")
    
    # Use the first TOML file found (typically config.toml)
    return toml_files[0]


def validate_subdirectories(main_dir_path):
    """
    Validate that all subdirectories contain required data files.
    
    Args:
        main_dir_path (str): Absolute path to the main directory
        
    Returns:
        list: List of valid subdirectory paths
        
    Raises:
        FileNotFoundError: If any subdirectory is missing required files
    """
    subdirs = []
    required_files = ["train.pt", "dev.pt", "test.pt"]
    
    # Get all subdirectories (excluding files)
    for item in sorted(os.listdir(main_dir_path)):
        item_path = os.path.join(main_dir_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            subdirs.append(item_path)
    
    # Validate each subdirectory
    for sub_dir in subdirs:
        data_dir = os.path.join(sub_dir, "data")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Subdirectory '{os.path.basename(sub_dir)}' is missing data/ folder. Expected: data/train.pt, data/dev.pt, data/test.pt")
        
        missing_files = []
        for req_file in required_files:
            file_path = os.path.join(data_dir, req_file)
            if not os.path.exists(file_path):
                missing_files.append(req_file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Subdirectory '{os.path.basename(sub_dir)}' is missing required data files: {', '.join(missing_files)}. "
                f"Expected: data/train.pt, data/dev.pt, data/test.pt"
            )
    
    return subdirs


def make_toml(sub_dir, main_toml_content):
    """
    Create modified TOML content for a specific subdirectory.
    
    Args:
        sub_dir (str): Path to the current subdirectory being processed
        main_toml_content (dict): Parsed content of the main TOML file
        
    Returns:
        dict: Modified TOML content object
    """
    # Create a deep copy of the main TOML content
    import copy
    modified_toml = copy.deepcopy(main_toml_content)
    
    # Get absolute paths
    main_dir = os.path.dirname(sub_dir)
    sub_dir_name = os.path.basename(sub_dir)
    
    # Update created_date to today's date
    today = datetime.now().strftime("%Y-%m-%d")
    if "meta_data" not in modified_toml:
        raise NameError(f"The main .toml is missing a '[meta_data]'")
    modified_toml["meta_data"]["created_date"] = today

    if modified_toml["meta_data"]["run_type"].lower() != "static":
        raise ValueError(f"This script is for k-fold and it expects run_type == 'static' but got:{modified_toml["meta_data"]["run_type"]}")
    
    # Update experiment directory
    expt_dir = os.path.join(sub_dir, "runs")
    modified_toml["meta_data"]["expt_dir"] = expt_dir
    
    # Update data paths in [static.data] section
    if "static" not in modified_toml:
        raise NameError(f"The main .toml is missing a '[static]'")
    if "data" not in modified_toml["static"]:
        raise NameError(f"The main .toml is missing a '[data]'")
    
    if modified_toml["static"]["num_runs"] > 1:
        print("\n\n")
        warnings.warn(
            f"{'\033[31m'}The 'num_runs' is > 1 ({modified_toml['static']['num_runs']}) "
            f"this means we will run EACH fold {modified_toml['static']['num_runs']} times.\n"
            f"Are you sure you want to do this?{'\033[0m'}\n\n"
        )

    
    data_section = modified_toml["static"]["data"]
    data_section["train_path"] = os.path.join(sub_dir, "data")
    data_section["dev_path"] = os.path.join(sub_dir, "data")
    data_section["test_path"] = os.path.join(sub_dir, "data")
    
    return modified_toml


def save_toml(sub_dir, toml_content):
    """
    Save the TOML content to the subdirectory.
    
    Args:
        sub_dir (str): Path to the subdirectory
        toml_content (dict): Modified TOML content to save
    """
    toml_path = os.path.join(sub_dir, "config.toml")
    
    try:
        with open(toml_path, 'w') as f:
            toml.dump(toml_content, f)
        print(f"Saved TOML configuration to: {toml_path}")
    except Exception as e:
        raise IOError(f"Failed to save TOML file to '{toml_path}': {str(e)}")


def run_experiment(sub_dir):
    """
    Run experiment for a specific subdirectory.
    
    Args:
        sub_dir (str): Path to the subdirectory
        
    Raises:
        Exception: If ExperimentBuilder or Expt_Runner fails
    """
    sub_dir_name = os.path.basename(sub_dir)
    toml_file_path = os.path.join(sub_dir, "config.toml")
    
    try:
        # Create ExperimentBuilder
        print(f"Building experiment for subdirectory: {sub_dir_name}")
        expt_builder = ExperimentBuilder(toml_file_path=toml_file_path)
        expt_builder.build_experiment_runs()
        
        # Create and run Expt_Runner
        print(f"Running experiment for subdirectory: {sub_dir_name}")
        runner = Expt_Runner(
            expt_dir=expt_builder.get_experiment_dir(),
            sub_runs_list=expt_builder.get_sub_runs_list(),
            run_on_gpu=expt_builder.get_run_on_gpu()
        )
        runner.run()
        
        print(f"Completed experiment for subdirectory: {sub_dir_name}")
        
    except Exception as e:
        if "ExperimentBuilder" in str(e) or "build_experiment_runs" in str(e):
            raise Exception(f"ExperimentBuilder failed for subdirectory '{sub_dir_name}': {str(e)}")
        else:
            raise Exception(f"Expt_Runner failed for subdirectory '{sub_dir_name}': {str(e)}")


def k_fold_runner(main_dir_path):
    """
    Main function to process multiple subdirectories for ML experiments.
    
    Args:
        main_dir_path (str): Absolute path to the main directory
    """
    try:
        print(f"Starting K-Fold Runner for directory: {main_dir_path}")
        
        # Phase 0: Validation
        print("Phase 0: Validating directory structure...")
        main_toml_path = validate_main_directory(main_dir_path)
        subdirs = validate_subdirectories(main_dir_path)
        
        # Load main TOML content
        try:
            with open(main_toml_path, 'r') as f:
                main_toml_content = toml.load(f)
        except toml.TomlDecodeError as e:
            raise ValueError(f"TOML file is malformed: {str(e)}")
        
        print(f"Found {len(subdirs)} valid subdirectories")
        
        # Phase 1: TOML Generation
        print("\nPhase 1: Generating TOML configurations...")
        for sub_dir in subdirs:
            sub_dir_name = os.path.basename(sub_dir)
            print(f"Processing TOML for: {sub_dir_name}")
            
            modified_toml = make_toml(sub_dir, main_toml_content)
            save_toml(sub_dir, modified_toml)
        
        # Phase 2: Experiment Execution
        print("\nPhase 2: Running experiments...")
        for sub_dir in subdirs:
            sub_dir_name = os.path.basename(sub_dir)
            print(f"\n{'='*50}")
            print(f"Starting experiment for: {sub_dir_name}")
            print(f"{'='*50}")
            
            run_experiment(sub_dir)
        
        print(f"\n{'='*50}")
        print("K-Fold Runner completed successfully!")
        print(f"Processed {len(subdirs)} subdirectories")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python k-fold_runner.py <main_dir_path>", file=sys.stderr)
        sys.exit(1)
    
    main_dir_path = os.path.abspath(sys.argv[1])
    k_fold_runner(main_dir_path)