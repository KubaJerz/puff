# Hyperparameter Analysis Report
**/home/kuba/projects/puff/paper00/experiments/01/runs/00findhyperparams**

## Best Models

| Rank | Model Name | F1 Score | Hyperparameters | Notes |
|------|------------|----------|-----------------|-------|
| 1 | 21 | 0.7911 | lr=0.0001, batch_size=256 | 🏆 Best Overall, 📊 Best Dev |
| 2 | 8 | 0.7871 | lr=0.0001, batch_size=256 |  |
| 3 | 4 | 0.7867 | lr=0.001, batch_size=256 |  |
| 4 | 0 | 0.7866 | lr=0.0001, batch_size=128 |  |
| 5 | 2 | 0.7862 | lr=0.0001, batch_size=256 |  |

## Worst Models

| Rank | Model Name | F1 Score | Hyperparameters |
|------|------------|----------|------------------|
| -1 | 10 | 0.0000 | lr=0.0003, batch_size=16 |
| -2 | 14 | 0.2904 | lr=0.001, batch_size=16 |
| -3 | 22 | 0.6565 | lr=0.0001, batch_size=16 |
| -4 | 9 | 0.7055 | lr=0.001, batch_size=32 |
| -5 | 17 | 0.7429 | lr=0.0003, batch_size=32 |

## Hyperparameter Impact Analysis

### Lr

- **0.0001** (avg F1: 0.7703, count: 12)
- **0.001** (avg F1: 0.7187, count: 9)
- **0.0003** (avg F1: 0.5753, count: 4)

### Batch_Size

- **256** (avg F1: 0.7838, count: 7)
- **128** (avg F1: 0.7816, count: 7)
- **64** (avg F1: 0.7811, count: 5)
- **32** (avg F1: 0.7340, count: 3)
- **16** (avg F1: 0.3156, count: 3)

