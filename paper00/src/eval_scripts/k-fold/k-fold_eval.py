#!/usr/bin/env python3
"""
K-Fold Cross-Validation Evaluation Script

Performs k-fold cross-validation evaluation across multiple model subdirectories,
calculates aggregate metrics, and generates visualizations.

Usage:
    python k-fold-eval.py <main_directory_path> -t <model_type> [--colors <theme>] [--no-gpu] [--model_outputs_logits] [-bs <batch_size>]
"""

import os
import sys
import argparse
import importlib
from pathlib import Path
import torch
import toml
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import BinaryF1Score
from torch.utils.data import TensorDataset, DataLoader


class ColorTheme:
    """Color themes for visualization styling"""
    
    THEMES = {
        'default': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'background': '#F8F9FA',
            'text': '#212529',
            'grid': '#E9ECEF',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545'
        },
        'academic': {
            'primary': '#1F4E79',
            'secondary': '#8B4513',
            'accent': '#B22222',
            'background': '#FFFFFF',
            'text': '#000000',
            'grid': '#D3D3D3',
            'success': '#006400',
            'warning': '#DAA520',
            'danger': '#8B0000'
        },
        'modern': {
            'primary': '#6366F1',
            'secondary': '#EC4899',
            'accent': '#10B981',
            'background': '#F9FAFB',
            'text': '#111827',
            'grid': '#E5E7EB',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626'
        }
    }
    
    @classmethod
    def get_theme(cls, theme_name):
        """Get color theme by name"""
        if theme_name not in cls.THEMES:
            raise ValueError(f"Unknown theme: {theme_name}. Available: {list(cls.THEMES.keys())}")
        return cls.THEMES[theme_name]


def setup_criterion(crit_config):
    """
    Set up loss criterion from configuration
    
    Args:
        crit_config (dict): Criterion configuration from TOML
        
    Returns:
        torch.nn.Module: Instantiated loss criterion
    """
    crit_class_str = crit_config["criterion"]
    crit_params = crit_config.get("criterion_params", {})
    
    if '/' in crit_class_str:  # Full path format
        module_dir = os.path.dirname(crit_class_str)
        module_name, loss_class_name = os.path.basename(crit_class_str).rsplit('.', 1)
        sys.path.insert(0, module_dir)
        module = importlib.import_module(module_name)
        crit_class = getattr(module, loss_class_name)
    else:
        # Handle other formats - assume it's a torch.nn module
        if hasattr(torch.nn, crit_class_str):
            crit_class = getattr(torch.nn, crit_class_str)
        else:
            raise ValueError(f"Unknown criterion format: {crit_class_str}")
    
    return crit_class(**crit_params)


def create_dataloader(X, y, batch_size):
    """
    Create DataLoader from tensor data
    
    Args:
        X: Input tensor data
        y: Target tensor data  
        batch_size (int): Batch size for DataLoader
        device: PyTorch device
        
    Returns:
        torch.utils.data.DataLoader: DataLoader object
    """
    # Ensure data is on CPU for DataLoader creation
    X_cpu = X.cpu() if X.is_cuda else X
    y_cpu = y.cpu() if y.is_cuda else y
    
    dataset = TensorDataset(X_cpu, y_cpu)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader


def run_model_based_on_type_batched(model, dataloader, model_type, model_outputs_logits, device):
    """
    Run model evaluation based on model type using batched processing
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing batched data
        model_type (str): Type of model ('binary_seg', etc.)
        model_outputs_logits (bool): Whether model outputs logits
        device: PyTorch device
        
    Returns:
        tuple: (all_predictions, all_logits, all_targets)
    """
    model.eval()
    model.to(device)
    all_predictions = []
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Get model output
            logits = model(batch_X)
            
            if model_type == 'binary_seg':
                if model_outputs_logits:
                    predictions = torch.nn.functional.sigmoid(logits)
                else:
                    predictions = logits
                predictions = (predictions > 0.5).long().squeeze().flatten()
            else:
                raise ValueError(f"For type: {model_type} we dont know how to format outputs. Please add in run_model_based_on_type_batched")
            
            # Store results (move to CPU to save GPU memory)
            all_predictions.append(predictions.cpu())
            all_logits.append(logits.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_logits, all_targets


def calc_metrics_based_on_type_batched(all_targets, all_predictions, all_logits, criterion, model_type, device):
    """
    Calculate metrics from accumulated batch results
    
    Args:
        all_targets: Concatenated target tensors
        all_predictions: Concatenated prediction tensors
        all_logits: Concatenated logit tensors
        criterion: Loss function
        model_type (str): Type of model
        device: PyTorch device
        
    Returns:
        tuple: (f1_score, average_loss)
    """
    if model_type == 'binary_seg':
        # Move tensors to device for loss calculation
        all_logits_device = all_logits.to(device)
        all_targets_device = all_targets.to(device)
        
        # Calculate loss on full dataset
        loss = criterion(all_logits_device, all_targets_device)
        
        # Calculate F1 score on full dataset (can use CPU tensors)
        f1_metric = BinaryF1Score()
        f1_score = f1_metric(all_predictions, all_targets.flatten())
    else:
        raise ValueError(f"For type: {model_type} we dont know how to format outputs. Please add in calc_metrics_based_on_type_batched")
    
    return f1_score.item(), loss.item()


def load_model_from_config(config, model_file, device):
    """
    Load model from configuration and checkpoint file
    
    Args:
        config (dict): Configuration dictionary
        model_file (Path): Path to model checkpoint
        device: PyTorch device
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Dynamic model loading
    model_path = config['static']['model']['model_path']
    model_hyperparams = config['static']['model'].copy()
    del model_hyperparams['model_path']

    # Import model module
    module_dir = os.path.dirname(model_path)
    module_name = os.path.basename(model_path).replace('.py', '')
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    model_class = getattr(module, 'Model')

    # Instantiate and load model
    model = model_class(**model_hyperparams)
    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    return model


def evaluate_fold(subdirectory, model_type, model_outputs_logits, device, batch_size):
    """
    Evaluate a single fold using batch processing
    
    Args:
        subdirectory (Path): Path to fold subdirectory
        model_type (str): Type of model
        model_outputs_logits (bool): Whether model outputs logits
        device: PyTorch device
        batch_size (int): Batch size for evaluation
        
    Returns:
        tuple: (fold_name, f1_score, loss_value) or None if evaluation fails
    """
    fold_name = subdirectory.name
    
    try:
        # Check required files
        config_file = subdirectory / 'config.toml'
        test_data_file = subdirectory / 'data' / 'test.pt'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Missing required file: {config_file}")
        if not test_data_file.exists():
            raise FileNotFoundError(f"Missing required file: {test_data_file}")
        
        # Find runs directory and model file
        runs_dir = subdirectory / 'runs'
        if not runs_dir.exists():
            raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
        
        run_subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_subdirs:
            raise FileNotFoundError(f"No run directories found in {runs_dir}")
        
        # Use the first (or only) run directory found
        run_dir = run_subdirs[0]
        
        # if multiple runs they will be 0,1,2,etc.
        total_f1 = 0.0
        total_loss = 0.0

        for run_num in range(len(os.listdir(run_dir))):
            if not os.path.isdir(run_dir / f"{run_num}"):
                continue

            model_file = run_dir / f"{run_num}" / 'best_dev_model.pt'

            if not model_file.exists():
                raise FileNotFoundError(f"Missing required file: {model_file}")
            
            # Load configuration
            try:
                config = toml.load(config_file)
            except Exception as e:
                raise Exception(f"Failed to parse config.toml in {subdirectory}: {str(e)}")
            
            # Load model
            try:
                model = load_model_from_config(config, model_file, device)
            except Exception as e:
                raise Exception(f"Failed to load model from {subdirectory}: {str(e)}")
            
            # Load test data
            try:
                X, y = torch.load(test_data_file, map_location='cpu', weights_only=True)
                # Create DataLoader for batched processing
                dataloader = create_dataloader(X, y, batch_size)
            except Exception as e:
                raise Exception(f"Failed to load test data from {subdirectory}: {str(e)}")
            
            # Setup criterion
            try:
                criterion = setup_criterion(config['static']['criterion'])
            except Exception as e:
                raise Exception(f"Failed to setup criterion from {subdirectory}: {str(e)}")
            
            # Run evaluation with batching
            try:
                all_preds, all_logits, all_targets = run_model_based_on_type_batched(
                    model, dataloader, model_type, model_outputs_logits, device
                )
                f1_score, loss_value = calc_metrics_based_on_type_batched(
                    all_targets, all_preds, all_logits, criterion, model_type, device
                )
                total_f1 += f1_score
                total_loss += loss_value
            except Exception as e:
                raise Exception(f"Failed to evaluate model in {subdirectory}: {str(e)}")
        
        avg_f1_score, avg_loss_value = total_f1 / len(os.listdir(run_dir)), total_loss / len(os.listdir(run_dir))

        return fold_name, avg_f1_score, avg_loss_value

            
    except Exception as e:
        print(f"Error evaluating {fold_name}: {str(e)}")
        return None


def create_visualizations(results, main_dir, color_theme_name):
    """
    Create boxplot visualizations of the results
    
    Args:
        results (list): List of (fold_name, f1_score, loss_value) tuples
        main_dir (Path): Main directory path
        color_theme_name (str): Name of color theme to use
    """
    # Create figures directory
    figures_dir = main_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Extract scores and losses
    f1_scores = [result[1] for result in results]
    losses = [result[2] for result in results]
    
    # Get color theme
    theme = ColorTheme.get_theme(color_theme_name)
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = theme['background']
    plt.rcParams['axes.facecolor'] = theme['background']
    plt.rcParams['text.color'] = theme['text']
    plt.rcParams['axes.labelcolor'] = theme['text']
    plt.rcParams['xtick.color'] = theme['text']
    plt.rcParams['ytick.color'] = theme['text']
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # F1 scores boxplot
    box1 = ax1.boxplot(f1_scores, patch_artist=True,
                       boxprops=dict(facecolor=theme['primary'], alpha=0.7),
                       medianprops=dict(color=theme['accent'], linewidth=2),
                       whiskerprops=dict(color=theme['text']),
                       capprops=dict(color=theme['text']),
                       flierprops=dict(marker='o', markerfacecolor=theme['danger'], 
                                     markeredgecolor=theme['danger'], alpha=0.6))
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Best Dev F1 Scores', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, color=theme['grid'])
    ax1.set_xticklabels(['F1 Scores'])
    
    # Loss boxplot
    box2 = ax2.boxplot(losses, patch_artist=True,
                       boxprops=dict(facecolor=theme['secondary'], alpha=0.7),
                       medianprops=dict(color=theme['accent'], linewidth=2),
                       whiskerprops=dict(color=theme['text']),
                       capprops=dict(color=theme['text']),
                       flierprops=dict(marker='o', markerfacecolor=theme['danger'], 
                                     markeredgecolor=theme['danger'], alpha=0.6))
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Best Dev Losses', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, color=theme['grid'])
    ax2.set_xticklabels(['Losses'])
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = figures_dir / 'performance_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor=theme['background'], edgecolor='none')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")


def print_results(results):
    """
    Print formatted results to console
    
    Args:
        results (list): List of (fold_name, f1_score, loss_value) tuples
    """
    if not results:
        print("No results to display.")
        return
    
    # Sort by F1 score (descending)
    f1_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("\n=== F1 SCORES (Best to Worst) ===")
    for fold_name, f1_score, _ in f1_sorted:
        print(f"{fold_name}: {f1_score:.4f}")
    
    avg_f1 = np.mean([result[1] for result in results])
    print(f"\nAverage F1 Score: {avg_f1:.4f}")
    
    # Sort by loss (ascending - lower is better)
    loss_sorted = sorted(results, key=lambda x: x[2])
    
    print("\n=== LOSSES (Best to Worst) ===")
    for fold_name, _, loss_value in loss_sorted:
        print(f"{fold_name}: {loss_value:.4f}")
    
    avg_loss = np.mean([result[2] for result in results])
    print(f"\nAverage Loss: {avg_loss:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='K-Fold Cross-Validation Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('main_directory_path', type=str,
                       help='Path to the main directory containing model subdirectories')
    parser.add_argument('-t', '--model_type', required=True,
                       choices=['binary_seg'],
                       help='Type of model to format output properly')
    parser.add_argument('--colors', default='default',
                       choices=['default', 'academic', 'modern'],
                       help='Color theme for visualizations (default: default)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU-only evaluation')
    parser.add_argument('--model_outputs_logits', action='store_true',
                       help='If the model outputs logits and we need to sigmoid or softmax')
    parser.add_argument('-bs', '--batch-size', type=int, default=2048,
                       help='Batch size for model evaluation (default: 2048)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Convert main directory to Path object
    main_dir = Path(args.main_directory_path)
    if not main_dir.exists() or not main_dir.is_dir():
        print(f"Error: Main directory does not exist or is not a directory: {main_dir}")
        sys.exit(1)
    
    # Find fold subdirectories
    fold_dirs = []
    for item in main_dir.iterdir():
        if item.is_dir() and item.name.startswith('fold-'):
            fold_dirs.append(item)
    
    if not fold_dirs:
        print(f"Error: No fold subdirectories found in {main_dir}")
        sys.exit(1)
    
    # Sort fold directories by name
    fold_dirs.sort(key=lambda x: x.name)
    
    print(f"Found {len(fold_dirs)} fold directories:")
    for fold_dir in fold_dirs:
        print(f"  - {fold_dir.name}")
    
    # Evaluate each fold
    results = []
    print(f"\nEvaluating folds with batch size {args.batch_size}...")
    
    for fold_dir in fold_dirs:
        print(f"Processing {fold_dir.name}...", end=" ")
        result = evaluate_fold(
            fold_dir, 
            args.model_type, 
            args.model_outputs_logits, 
            device,
            args.batch_size
        )
        if result:
            results.append(result)
            print(f"✓ F1: {result[1]:.4f}, Loss: {result[2]:.4f}")
        else:
            print(f"{'\033[31m'}✗{'\033[0m'} Failed")
    
    if not results:
        print("\nError: No successful evaluations. Exiting.")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    # Create visualizations
    try:
        create_visualizations(results, main_dir, args.colors)
    except Exception as e:
        print(f"Warning: Failed to create visualizations: {str(e)}")
    
    print(f"\nEvaluation complete. Processed {len(results)}/{len(fold_dirs)} folds successfully.")


if __name__ == '__main__':
    main()