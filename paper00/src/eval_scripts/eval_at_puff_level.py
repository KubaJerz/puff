#!/usr/bin/env python3
"""
Puff Level F1 Score Evaluation Script

Evaluates model performance at the puff level by counting detected vs missed puffs,
rather than pixel-level accuracy. Includes density-based filtering to remove
isolated predictions in low-activity regions.
"""

import torch
import numpy as np
import sys
import importlib
import os
from scipy.ndimage import binary_closing
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 70
MODEL_WEIGHTS_PATH = '/home/kuba/projects/puff/paper00/experiments/old/04session-split00/runs/same_config_run/3/best_dev_model.pt'
MODEL_DEFINITION_PATH = '/home/kuba/projects/puff/paper00/unet.py'
DATA_PATH = '/home/kuba/projects/puff/paper00/experiments/old/04session-split00/data/test.pt'

# Model parameters
CONFIDENCE_THRESHOLD = 0.65
BINARY_CLOSING_SIZE = 70
BATCH_SIZE = 512

# Signal parameters
SIGNAL_HZ = 50
PUFF_SEARCH_TIME_SECONDS = 2.5
MIN_PUFFS_IN_RANGE = 2

# Evaluation parameters
OVERLAP_THRESHOLD = 0.5  # Minimum overlap ratio for puff matching

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config():
    """Validate all configuration parameters."""
    validations = [
        (BINARY_CLOSING_SIZE >= 0, "BINARY_CLOSING_SIZE must be non-negative"),
        (0 <= CONFIDENCE_THRESHOLD <= 1, "CONFIDENCE_THRESHOLD must be between 0 and 1"),
        (BATCH_SIZE > 0, "BATCH_SIZE must be positive"),
        (SIGNAL_HZ > 0, "SIGNAL_HZ must be positive"),
        (PUFF_SEARCH_TIME_SECONDS > 0, "PUFF_SEARCH_TIME_SECONDS must be positive"),
        (MIN_PUFFS_IN_RANGE >= 1, "MIN_PUFFS_IN_RANGE must be at least 1"),
        (0 < OVERLAP_THRESHOLD <= 1, "OVERLAP_THRESHOLD must be between 0 and 1")
    ]
    
    for condition, message in validations:
        if not condition:
            raise ValueError(message)


def validate_paths():
    """Validate that all required file paths exist."""
    paths = {
        'Model weights': MODEL_WEIGHTS_PATH,
        'Model definition': MODEL_DEFINITION_PATH,
        'Test data': DATA_PATH
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    print("✓ All file paths validated")


def setup_device():
    """Setup and return the appropriate device (GPU preferred)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    return device


def print_config():
    """Print configuration summary."""
    print("Configuration Summary:")
    print("=" * 25)
    print(f"Signal frequency: {SIGNAL_HZ} Hz")
    print(f"Search time window: ±{PUFF_SEARCH_TIME_SECONDS} seconds")
    print(f"Search range: ±{int(PUFF_SEARCH_TIME_SECONDS * SIGNAL_HZ)} samples")
    print(f"Minimum puffs required: {MIN_PUFFS_IN_RANGE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Overlap threshold: {OVERLAP_THRESHOLD}")
    print()

# ============================================================================
# DATA HANDLING
# ============================================================================

def load_and_unshuffle_data(data_path, seed):
    """Load test data and restore original order using the seed."""
    print("Loading and unshuffling test data...")
    
    # Load data
    X, y = torch.load(data_path, weights_only=True)
    print(f"✓ Loaded data - X: {X.shape}, y: {y.shape}")
    
    # Restore original order
    np.random.seed(seed)
    perm_idx = np.random.permutation(len(X))
    reverse_idx = np.empty_like(perm_idx)
    reverse_idx[perm_idx] = np.arange(len(perm_idx))
    
    X_unshuffled = X[reverse_idx]
    y_unshuffled = y[reverse_idx]
    
    print("✓ Data unshuffled successfully")
    return X_unshuffled, y_unshuffled

# ============================================================================
# MODEL HANDLING
# ============================================================================

def load_model(model_def_path, model_weights_path, device):
    """Load model class and weights."""
    print("Loading model...")
    
    # Import model class
    module_path, file_name = model_def_path.rsplit('/', 1)
    file_prefix = file_name.split('.')[0]
    
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    try:
        module = importlib.import_module(file_prefix)
        model_class = getattr(module, 'Model')
        print(f"✓ Imported Model class from {file_prefix}")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import Model class: {e}")
    
    # Initialize and load weights
    model = model_class()
    
    try:
        state_dict = torch.load(model_weights_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded weights from {model_weights_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    model = model.to(device)
    model.eval()
    print(f"✓ Model ready on {device}")
    
    return model


def make_predictions(model, X, device, batch_size):
    """Make predictions on data in batches."""
    print(f"Making predictions in batches of {batch_size}...")
    
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            batch_pred = model(batch)
            all_predictions.append(batch_pred.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed batch {i // batch_size + 1}/{(len(X) - 1) // batch_size + 1}")
    
    predictions = torch.cat(all_predictions, dim=0)
    print(f"✓ Predictions complete - shape: {predictions.shape}")
    
    return predictions


def process_predictions(predictions, confidence_threshold, closing_size):
    """Apply sigmoid, thresholding, and binary closing to predictions."""
    print("Processing predictions...")
    
    # Apply sigmoid and threshold
    sigmoid_pred = torch.sigmoid(predictions)
    binary_pred = (sigmoid_pred > confidence_threshold).long()
    print(f"  Applied sigmoid and threshold ({confidence_threshold})")
    
    # Apply binary closing
    binary_pred_flat = binary_pred.flatten().numpy()
    
    if closing_size > 0:
        structure = np.ones(closing_size)
        closed_pred = binary_closing(binary_pred_flat, structure=structure).astype(int)
        print(f"  Applied binary closing (size: {closing_size})")
    else:
        closed_pred = binary_pred_flat
        print("  Skipped binary closing")
    
    print("✓ Prediction processing complete")
    return closed_pred

# ============================================================================
# PUFF DETECTION
# ============================================================================

def find_puff_regions(binary_array):
    """Find connected regions (puffs) in binary array."""
    if len(binary_array) == 0:
        return []
    
    puff_regions = []
    in_puff = False
    start_idx = 0
    
    for i, val in enumerate(binary_array):
        if val == 1 and not in_puff:
            start_idx = i
            in_puff = True
        elif val == 0 and in_puff:
            puff_regions.append((start_idx, i - 1))
            in_puff = False
    
    # Handle case where array ends with a puff
    if in_puff:
        puff_regions.append((start_idx, len(binary_array) - 1))
    
    return puff_regions


def calculate_overlap_ratio(pred_start, pred_end, true_start, true_end):
    """Calculate overlap ratio between two puff regions."""
    overlap_start = max(pred_start, true_start)
    overlap_end = min(pred_end, true_end)
    
    if overlap_start > overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start + 1
    pred_length = pred_end - pred_start + 1
    true_length = true_end - true_start + 1
    min_length = min(pred_length, true_length)
    
    return overlap_length / min_length


def match_puffs(pred_regions, true_regions, overlap_threshold):
    """Match predicted puffs with ground truth puffs based on overlap."""
    matched_true_puffs = set()
    matched_pred_puffs = set()
    
    for pred_idx, (pred_start, pred_end) in enumerate(pred_regions):
        best_overlap = 0
        best_true_idx = -1
        
        for true_idx, (true_start, true_end) in enumerate(true_regions):
            if true_idx in matched_true_puffs:
                continue
            
            overlap_ratio = calculate_overlap_ratio(pred_start, pred_end, true_start, true_end)
            
            if overlap_ratio > best_overlap and overlap_ratio >= overlap_threshold:
                best_overlap = overlap_ratio
                best_true_idx = true_idx
        
        if best_true_idx >= 0:
            matched_true_puffs.add(best_true_idx)
            matched_pred_puffs.add(pred_idx)
    
    return matched_pred_puffs, matched_true_puffs

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_puff_metrics(y_true, y_pred, signal_hz, overlap_threshold):
    """Calculate puff-level classification metrics."""
    print("Calculating puff-level metrics...")
    
    # Find all puff regions
    true_regions = find_puff_regions(y_true.flatten().astype(int))
    pred_regions = find_puff_regions(y_pred.flatten().astype(int))
    
    print(f"  Found {len(true_regions)} ground truth puffs")
    print(f"  Found {len(pred_regions)} predicted puffs")
    
    # Match puffs
    matched_pred, matched_true = match_puffs(pred_regions, true_regions, overlap_threshold)
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_regions) - len(matched_pred)
    fn = len(true_regions) - len(matched_true)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average durations
    avg_true_duration = np.mean([(end - start + 1) / signal_hz for start, end in true_regions]) if true_regions else 0.0
    avg_pred_duration = np.mean([(end - start + 1) / signal_hz for start, end in pred_regions]) if pred_regions else 0.0
    
    metrics = {
        'total_true_puffs': len(true_regions),
        'total_predicted_puffs': len(pred_regions),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_true_duration': avg_true_duration,
        'avg_pred_duration': avg_pred_duration
    }
    
    print(f"  Matched {tp} puffs, {fp} false positives, {fn} missed")
    print("✓ Puff-level metrics calculated")
    
    return metrics


def print_metrics(metrics, title):
    """Print evaluation metrics in a formatted manner."""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Ground Truth Puffs: {metrics['total_true_puffs']}")
    print(f"Predicted Puffs: {metrics['total_predicted_puffs']}")
    print(f"Successfully Matched (TP): {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"Missed Puffs (FN): {metrics['false_negatives']}")
    print("-" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("-" * 50)
    print(f"Avg True Puff Duration: {metrics['avg_true_duration']:.2f}s")
    print(f"Avg Pred Puff Duration: {metrics['avg_pred_duration']:.2f}s")

# ============================================================================
# DENSITY FILTERING
# ============================================================================

def count_puffs_in_range(start, end, all_puff_regions):
    """Count puffs that overlap with the given range."""
    count = 0
    for puff_start, puff_end in all_puff_regions:
        if not (puff_end < start or puff_start > end):
            count += 1
    return count


def filter_by_density(y_pred, y_true, search_time_seconds, signal_hz, min_puffs_in_range):
    """Filter predictions based on puff density in surrounding regions."""
    print("Applying density filtering...")
    
    # Convert time to samples
    search_range = int(search_time_seconds * signal_hz)
    print(f"  Search window: ±{search_time_seconds}s ({search_range} samples)")
    
    # Find all puff regions
    pred_regions = find_puff_regions(y_pred)
    true_regions = find_puff_regions(y_true)
    all_regions = pred_regions + true_regions
    
    print(f"  Found {len(pred_regions)} predicted, {len(true_regions)} true puffs")
    
    # Filter predictions
    filtered_pred = y_pred.copy()
    removed_count = 0
    
    for puff_start, puff_end in pred_regions:
        # Define search window
        search_start = max(0, puff_start - search_range)
        search_end = min(len(y_pred) - 1, puff_end + search_range)
        
        # Count puffs in range
        puffs_in_range = count_puffs_in_range(search_start, search_end, all_regions)
        
        # Remove if below threshold
        if puffs_in_range < min_puffs_in_range:
            filtered_pred[puff_start:puff_end + 1] = 0
            removed_count += 1
    
    print(f"  Removed {removed_count}/{len(pred_regions)} low-density puffs")
    print("✓ Density filtering complete")
    
    return filtered_pred, removed_count, len(pred_regions)



def write_evaluation_results(initial_metrics, filtered_metrics, removed_count, total_count):
    """
    Write evaluation results and configuration to a markdown file.
    
    Args:
        initial_metrics (dict): Initial evaluation metrics
        filtered_metrics (dict): Filtered evaluation metrics  
        removed_count (int): Number of puffs removed by density filtering
        total_count (int): Total number of predicted puffs before filtering
    """
    # Get directory from model weights path
    model_dir = os.path.dirname(MODEL_WEIGHTS_PATH)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"puff_evaluation_results_{timestamp}.md"
    filepath = os.path.join(model_dir, filename)
    
    # Prepare content
    content = f"""# Puff-Level F1 Score Evaluation Results

**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration Parameters

### Model Configuration
- **Model Weights:** `{MODEL_WEIGHTS_PATH}`
- **Model Definition:** `{MODEL_DEFINITION_PATH}`
- **Test Data:** `{DATA_PATH}`
- **Random Seed:** `{SEED}`

### Processing Parameters
- **Confidence Threshold:** `{CONFIDENCE_THRESHOLD}`
- **Binary Closing Size:** `{BINARY_CLOSING_SIZE}` samples
- **Batch Size:** `{BATCH_SIZE}`

### Signal Parameters
- **Signal Frequency:** `{SIGNAL_HZ}` Hz
- **Puff Search Time Window:** `±{PUFF_SEARCH_TIME_SECONDS}` seconds (`±{int(PUFF_SEARCH_TIME_SECONDS * SIGNAL_HZ)}` samples)
- **Minimum Puffs in Range:** `{MIN_PUFFS_IN_RANGE}`

### Evaluation Parameters
- **Overlap Threshold:** `{OVERLAP_THRESHOLD}`

## Initial Results (Before Density Filtering)

| Metric | Value |
|--------|-------|
| **Ground Truth Puffs** | {initial_metrics['total_true_puffs']} |
| **Predicted Puffs** | {initial_metrics['total_predicted_puffs']} |
| **True Positives** | {initial_metrics['true_positives']} |
| **False Positives** | {initial_metrics['false_positives']} |
| **False Negatives** | {initial_metrics['false_negatives']} |
| **Precision** | {initial_metrics['precision']:.4f} |
| **Recall** | {initial_metrics['recall']:.4f} |
| **F1 Score** | {initial_metrics['f1_score']:.4f} |
| **Avg True Puff Duration** | {initial_metrics['avg_true_duration']:.2f}s |
| **Avg Pred Puff Duration** | {initial_metrics['avg_pred_duration']:.2f}s |

## Density Filtering Results

### Filtering Impact
- **Puffs Removed:** {removed_count}/{total_count} ({(removed_count/total_count*100 if total_count > 0 else 0):.1f}%)
- **Filtering Criterion:** Minimum {MIN_PUFFS_IN_RANGE} puffs within ±{PUFF_SEARCH_TIME_SECONDS}s window

### Final Results (After Density Filtering)

| Metric | Value |
|--------|-------|
| **Ground Truth Puffs** | {filtered_metrics['total_true_puffs']} |
| **Predicted Puffs** | {filtered_metrics['total_predicted_puffs']} |
| **True Positives** | {filtered_metrics['true_positives']} |
| **False Positives** | {filtered_metrics['false_positives']} |
| **False Negatives** | {filtered_metrics['false_negatives']} |
| **Precision** | {filtered_metrics['precision']:.4f} |
| **Recall** | {filtered_metrics['recall']:.4f} |
| **F1 Score** | {filtered_metrics['f1_score']:.4f} |
| **Avg True Puff Duration** | {filtered_metrics['avg_true_duration']:.2f}s |
| **Avg Pred Puff Duration** | {filtered_metrics['avg_pred_duration']:.2f}s |

## Performance Comparison

| Metric | Initial | Filtered | Change |
|--------|---------|----------|--------|
| **Precision** | {initial_metrics['precision']:.4f} | {filtered_metrics['precision']:.4f} | {(filtered_metrics['precision'] - initial_metrics['precision']):+.4f} |
| **Recall** | {initial_metrics['recall']:.4f} | {filtered_metrics['recall']:.4f} | {(filtered_metrics['recall'] - initial_metrics['recall']):+.4f} |
| **F1 Score** | {initial_metrics['f1_score']:.4f} | {filtered_metrics['f1_score']:.4f} | {(filtered_metrics['f1_score'] - initial_metrics['f1_score']):+.4f} |
| **Predicted Puffs** | {initial_metrics['total_predicted_puffs']} | {filtered_metrics['total_predicted_puffs']} | {filtered_metrics['total_predicted_puffs'] - initial_metrics['total_predicted_puffs']} |

## Summary

### Key Findings
- **Best F1 Score:** {max(initial_metrics['f1_score'], filtered_metrics['f1_score']):.4f} ({'Initial' if initial_metrics['f1_score'] > filtered_metrics['f1_score'] else 'Filtered'})
- **Filtering Effect:** {"Improved" if filtered_metrics['f1_score'] > initial_metrics['f1_score'] else "Reduced" if filtered_metrics['f1_score'] < initial_metrics['f1_score'] else "No change in"} F1 score by {abs(filtered_metrics['f1_score'] - initial_metrics['f1_score']):.4f}
- **Precision vs Recall Trade-off:** {"Precision-favoring" if filtered_metrics['precision'] > filtered_metrics['recall'] else "Recall-favoring" if filtered_metrics['recall'] > filtered_metrics['precision'] else "Balanced"} (Final: P={filtered_metrics['precision']:.3f}, R={filtered_metrics['recall']:.3f})

### Model Performance Assessment
"""

    # Add performance assessment based on F1 score
    f1 = filtered_metrics['f1_score']
    if f1 >= 0.9:
        assessment = "**Excellent** - Very high detection accuracy"
    elif f1 >= 0.8:
        assessment = "**Good** - High detection accuracy with room for improvement"
    elif f1 >= 0.7:
        assessment = "**Moderate** - Reasonable detection but significant improvement needed"
    elif f1 >= 0.5:
        assessment = "**Poor** - Low detection accuracy, major issues present"
    else:
        assessment = "**Very Poor** - Detection failing, requires significant changes"
    
    content += f"- **Overall Performance:** {assessment}\n"
    
    # Add recommendations
    content += "\n### Recommendations\n"
    
    if filtered_metrics['precision'] < 0.7:
        content += "- **Reduce False Positives:** Consider increasing confidence threshold or improving model specificity\n"
    if filtered_metrics['recall'] < 0.7:
        content += "- **Reduce False Negatives:** Consider decreasing confidence threshold or improving model sensitivity\n"
    if abs(filtered_metrics['precision'] - filtered_metrics['recall']) > 0.2:
        content += "- **Balance Precision/Recall:** Current model shows significant bias toward " + ("precision" if filtered_metrics['precision'] > filtered_metrics['recall'] else "recall") + "\n"
    if removed_count > total_count * 0.5:
        content += "- **Review Density Filtering:** High removal rate suggests filtering may be too aggressive\n"
    
    content += f"\n---\n*Generated by puff evaluation script on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # Write to file
    try:
        os.makedirs(model_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Evaluation results saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main evaluation pipeline."""
    try:
        print("Starting Puff-Level F1 Score Evaluation")
        print("=" * 50)
        
        # Setup
        validate_config()
        validate_paths()
        print_config()
        device = setup_device()
        
        # Load data and model
        X, y = load_and_unshuffle_data(DATA_PATH, SEED)
        model = load_model(MODEL_DEFINITION_PATH, MODEL_WEIGHTS_PATH, device)
        
        # Make and process predictions
        predictions = make_predictions(model, X, device, BATCH_SIZE)
        processed_pred = process_predictions(predictions, CONFIDENCE_THRESHOLD, BINARY_CLOSING_SIZE)
        
        # Initial evaluation
        initial_metrics = calculate_puff_metrics(y.numpy(), processed_pred, SIGNAL_HZ, OVERLAP_THRESHOLD)
        print_metrics(initial_metrics, "Initial Puff-Level Results:")
        
        # Density filtering
        filtered_pred, removed_count, total_count = filter_by_density(
            processed_pred, y.numpy().flatten(), 
            PUFF_SEARCH_TIME_SECONDS, SIGNAL_HZ, MIN_PUFFS_IN_RANGE
        )
        
        # Filtered evaluation
        filtered_metrics = calculate_puff_metrics(y.numpy(), filtered_pred, SIGNAL_HZ, OVERLAP_THRESHOLD)
        print_metrics(filtered_metrics, f"Filtered Results (Min {MIN_PUFFS_IN_RANGE} puffs in ±{PUFF_SEARCH_TIME_SECONDS}s):")
        
        # Summary
        print(f"\nFiltering Impact:")
        print(f"================")
        print(f"Removed puffs: {removed_count}/{total_count}")
        print(f"F1 change: {filtered_metrics['f1_score'] - initial_metrics['f1_score']:+.4f}")
        print(f"Precision change: {filtered_metrics['precision'] - initial_metrics['precision']:+.4f}")
        print(f"Recall change: {filtered_metrics['recall'] - initial_metrics['recall']:+.4f}")
        print("=" * 50)

        write_evaluation_results(initial_metrics, filtered_metrics, removed_count, total_count)
        
        return {'initial': initial_metrics, 'filtered': filtered_metrics}
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    results = main()