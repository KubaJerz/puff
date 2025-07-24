#!/bin/bash

# dir-setup-k-fold.sh
# Creates a k-fold cross-validation directory structure for machine learning experiments

# Function to display usage information
usage() {
    echo "Usage: $0 <path> -n <number_of_folds>"
    echo "Example: $0 /path/to/experiments -n 5"
    exit 1
}

# Function to display error messages and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    usage
fi

# Parse arguments
TARGET_PATH="$1"
FLAG="$2"
NUM_FOLDS="$3"

# Validate flag
if [ "$FLAG" != "-n" ]; then
    usage
fi

# Validate number of folds is a positive integer
if ! [[ "$NUM_FOLDS" =~ ^[1-9][0-9]*$ ]]; then
    error_exit "Number of folds must be a positive integer"
fi

# Check if target path exists
if [ ! -d "$TARGET_PATH" ]; then
    error_exit "Path '$TARGET_PATH' does not exist"
fi

# Define k-fold directory path
KFOLD_DIR="$TARGET_PATH/k-fold"

# Check if k-fold directory already exists
if [ -d "$KFOLD_DIR" ]; then
    error_exit "Directory '$KFOLD_DIR' already exists"
fi

# Create main k-fold directory
echo "Creating k-fold directory structure at: $KFOLD_DIR"
mkdir "$KFOLD_DIR" || error_exit "Failed to create directory '$KFOLD_DIR'"
echo "Created main directory: $KFOLD_DIR"

# Create fold directories
echo "Created fold directories:"
for ((i=0; i<NUM_FOLDS; i++)); do
    FOLD_DIR="$KFOLD_DIR/fold-$i"
    DATA_DIR="$FOLD_DIR/data"
    RUNS_DIR="$FOLD_DIR/runs"
    
    # Create fold directory
    mkdir "$FOLD_DIR" || error_exit "Failed to create fold directory '$FOLD_DIR'"
    
    # Create data and runs subdirectories
    mkdir "$DATA_DIR" || error_exit "Failed to create data directory '$DATA_DIR'"
    mkdir "$RUNS_DIR" || error_exit "Failed to create runs directory '$RUNS_DIR'"
    
    echo "  - fold-$i (with data/ and runs/ subdirectories)"
done

echo "Successfully created k-fold structure with $NUM_FOLDS folds."
exit 0