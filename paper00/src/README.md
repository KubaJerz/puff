# How to use the scripts

First we should note the structure so then the way the scritps work will make sense. 
There are these things called "experiments" and each experiment can have sub-runs in it anywhere from 1 to N.
Each experiment will have a `data/` and `runs/` dir so all the runs will have used the same data but can have diffrent model, criterion, optimizer, etc.

Generaly for a run the model type will be the same but there can be diffrent hyperparams for each sub run. There are two types of runs. "*static*" and "*sweep*". The "*static*" will do keep every thing the same (hyperparmeters, etc.) between each sub run. The "*sweep*" will sample via `random.choice` from a set of params in a list in  `[sweep.search_space]`



But experiments can also be "*macro*" experiments which is just a dir that contains many dirs of related experiments. 

Example Structure:
```
experiments/
├── experiment_001/
│   ├── data/
│   └── runs/
|         ├──run00_transformer
|         |     ├──0
|         |     |   ├──metrics.png
|         |     |   ├──run_params.toml
|         |     |   ├──best_dev_metrics.json
|         |     |   └──best_dev_model.pt
|         |     ├──1
|         |     └──run00_transformer.toml
|         ├──run01_cnn
|         ├──run02_mlp
|         └──run03_lstm
|         
├── experiment_002/
│   ├── data/
│   └── runs/
├── macro_experiment_k_fold
│   ├── fold_001/
|   │   ├── data/
|   │   └── runs/
|   |       └── k-fold
|   |       |     └── 0
|   |       |        ├──metrics.png
|   |       |        ├──run_params.toml
|   |       |       ├──best_dev_metrics.json
|   |       |        └──best_dev_model.pt
|   |       └── config.toml
|   |
│   ├── fold_002/
|   │   ├── data/
|   │   └── runs/
│   └── fold_03/
|   │   ├── data/
|   │   └── runs/
└── ...
```


## How to run.
1. Setup dir structure for experiment
2. Setup `.toml` file
3. Setup `data/` dir
4. Setup run `.toml` file with the correct script

### 1. Setup dir structure
For an simple non "*macro experiment*" experiment you jsut need to make the `experiment_00` dir then also the `runs/` and `data/` dirs. And them specify the `.toml` as you would like

- For K-Fold
    - Use the `dir-setup-k-fold.sh` to setupt the dir
    - The `.toml` file needs to be in the  `macro_experimet` dir

- For K-Fold Finetune/Customize
    - No dir setup needed
    -  The `.toml` file path needs to be past to the script `  python k_fold_customizer.py config.toml`


### 2 `.toml setup`
#### Basic "static" .toml as fresh weights or loading weights
```toml
[meta_data]
run_type = "static"
name = "01tryHyperparamsFrom21"
description = "training on a few models with the same hyperparams as model 21 from run 00find hyperparams"
author = "Kuba"
created_date = "2025-07-18"
expt_dir = "/home/kuba/projects/puff/paper00/experiments/01/runs"
plot_freq = 25

[static]
num_runs = 10
epochs = 100
run_on_gpu = true
save_checkpoints = true

[static.model]
model_path = "/home/kuba/projects/puff/paper00/unet.py"
#model_weights = "path/to/old/weights/model.pt" (OPTIONAL)
in_channels = 3
#every thing other than "mode_path" and "weights_path" get unpacked in the model constructor


[static.optimizer]
optimizer = "torch.optim.Adam"
# optimizer_weights = "path/to/saved/optim_state.pt" (OPTIONAL)
lr = 0.0001

[static.criterion]
criterion = "/home/kuba/projects/puff/test/loss.DiceBCELoss"

[static.data]
train_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
dev_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
test_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
batch_size = 256
use_test = false

```

**NOTE:** The two way to specify a loss fuction or optimizer are: "path/to/custom/file.Classname" or from imported module already via "torch.optim.Adam"

#### Basic "sweep" .toml 

```toml
[meta_data]
run_type = "sweep" 
name = "00findhyperparams"
description = "training on sessions the goal is jsut better label model"
author = "Kuba"
created_date = "2025-07-17"
expt_dir = "/home/kuba/projects/puff/paper00/experiments/01/runs"
plot_freq = 25

[sweep]
# Sampling strategy: "grid_search", "random_search"
sampling_strategy = "random_search"
num_runs = 25  # For random search, ignored for grid search
epochs = 100  
run_on_gpu = true #this does nothing right now

[sweep.model]
model_path = "/home/kuba/projects/puff/paper00/unet.py"
in_channels = 3

[sweep.optimizer]
optimizer = "torch.optim.Adam"


[sweep.criterion]
criterion = "/home/kuba/projects/puff/test/loss.DiceBCELoss"

[sweep.data]
train_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
dev_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
test_path = "/home/kuba/projects/puff/paper00/experiments/01/data"
batch_size = "ToBeSampled"
use_test = false
#num_workers = 4

# Parameters to sweep - define ranges/options
[sweep.search_space]
hidden_dim = [64, 128, 256, 512]
dropout = [0.1, 0.2, 0.3, 0.5]
lr = [0.0001, 0.001, 0.0003]
batch_size = [16, 32, 64, 128, 256]
weight_decay = [0.0, 1e-5, 1e-4, 1e-3]
```

**NOTE:** For the `sweep.search_space` you might need to modify the `_apply_sampled_params` method in `experiment_builder.py` if the mapping as not been added before. 

#### k-fold .toml 

```toml
[meta_data]
run_type = "static" 

name = "k-fold"
description = """
We are testing k-fold"""
author = "Kuba"
created_date = "2025-07-22"
expt_dir = "will be fixed"
plot_freq = 25


[static]
num_runs = 1
epochs = 100
run_on_gpu = true #this does nothing right now 
save_checkpoints = true
patience = 10

[static.model]
model_path = "/home/kuba/projects/puff/paper00/unet.py"
in_channels = 3


[static.optimizer]
optimizer = "torch.optim.Adam"
lr = 0.0001

[static.criterion]
criterion = "/home/kuba/projects/puff/test/loss.WeightedCenterLoss"

[static.data]
train_path = "n/a" #this will be filled in by script
dev_path = "n/a" 
test_path = "n/a" 
batch_size = 256
use_test = false
```

**NOTE:** It will fill in the data path for each one

#### k-fold customizer/finetune .toml 

```toml
# K-Fold Customizer Parameters
input_kfold_dir = "/home/kuba/Desktop/k-fold"
name = "testing-the-logic"
expt_dir = "/home/kuba/projects/puff/test/experiments"
percent_old_train = 0.1
test_split_ratio = 0.333
use_optimizer_state = false

# Experiment Template Configuration
[meta_data]
run_type = "static" 
name = "template"  # Will be overridden to "customize"
expt_dir = "None"  # Will be filled in for each fold
plot_freq = 10

[static]
num_runs = 1
epochs = 100
run_on_gpu = true
save_checkpoints = true
patience = 15

```

### 3. Setup data
Create datasets using the appropriate data preparation script:
Basic experiments:

- Participant-level splits: python 00participant_level.py
- Session-level splits: python 01session_level.py
- K-fold cross-validation: python 02kfold_data_prep.py

Configuration:

- Update SAVE_DIR in each script to point to your experiment/data/ directory

Output: Each script saves PyTorch tensors (.pt files) and configuration (.toml files) ready for training.

### 4. Send the .toml file to the right script

For most scripts you need to just do this:
```
we
```

- For k-fold you need to run via:
    - `k-fold_runner.py`
- For f-fold finetune/customization run via:
    - `k_fold_customizer.py`

## How to eval

Looks at eval docs