# Eval Scripts


## Hyperparameter Investigator

### Usage

```bash
python hyperparam_investigator.py /path/to/experiment/run/directory
```

## Overview

This script analyzes hyperparameter sweep results by:

1. Reading the main experiment TOML file to get the `[sweep.search_space]` section
2. Extracting actual hyperparameter values from subdirectory TOML files based on predefined paths
3. Ranking models by performance and analyzing hyperparameter impact

## Important Note

The script **only analyzes hyperparameters defined in `[sweep.search_space]`** of the main TOML file. The paths to find these hyperparameters in subdirectory TOML files must be specified in the script's path mapping.

## Path Mapping Examples

The script maps parameter names to their nested paths in subdirectory TOML files:

- **`lr`** → `['optimizer', 'optimizer_params', 'lr']`
  - Finds: `[optimizer.optimizer_params].lr = 0.0001`

- **`batch_size`** → `['data', 'batch_size']`  
  - Finds: `[data].batch_size = 256`

To add new hyperparameters, update the `path_mappings` dictionary in the `_get_config_path_for_param()` function.