# Docs for k-fold folder

## dir-setup-k-fold

Overview
A bash script that creates a k-fold cross-validation directory structure for machine learning experiments.
Script Name
dir-setup-k-fold.sh
Parameters
* Path parameter: Target directory path where the k-fold structure will be created
* -n parameter: Number of folds (groups) to create
Usage

`./dirSetup-k-fold.sh /path/to/experiments -n 5`

Directory Structure Created

```
<provided-path>/
└── k-fold/
    ├── fold-0/
    │   ├── data/
    │   └── runs/
    ├── fold-1/
    │   ├── data/
    │   └── runs/
    ├── fold-2/
    │   ├── data/
    │   └── runs/
    └── ...
```