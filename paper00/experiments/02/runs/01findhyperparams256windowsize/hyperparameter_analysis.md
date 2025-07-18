# Hyperparameter Analysis Report
**/home/kuba/projects/puff/paper00/experiments/02/runs/01findhyperparams256windowsize**

## Best Models

| Rank | Model Name | F1 Score | Hyperparameters | Notes |
|------|------------|----------|-----------------|-------|
| 1 | 17 | 0.7884 | lr=0.0001, batch_size=128 | 🏆 Best Overall, 📊 Best Dev |
| 2 | 11 | 0.7875 | lr=0.001, batch_size=256 |  |
| 3 | 2 | 0.7851 | lr=0.0003, batch_size=256 |  |
| 4 | 22 | 0.7844 | lr=0.0003, batch_size=128 |  |
| 5 | 13 | 0.7843 | lr=0.0001, batch_size=128 |  |

## Worst Models

| Rank | Model Name | F1 Score | Hyperparameters |
|------|------------|----------|------------------|
| -1 | 16 | 0.0000 | lr=0.001, batch_size=16 |
| -2 | 5 | 0.0000 | lr=0.001, batch_size=16 |
| -3 | 15 | 0.5714 | lr=0.001, batch_size=32 |
| -4 | 21 | 0.6285 | lr=0.001, batch_size=32 |
| -5 | 8 | 0.6392 | lr=0.001, batch_size=32 |

## Hyperparameter Impact Analysis

### Lr

- **0.0003** (avg F1: 0.7670, count: 9)
- **0.0001** (avg F1: 0.7502, count: 6)
- **0.001** (avg F1: 0.5695, count: 10)

### Batch_Size

- **256** (avg F1: 0.7849, count: 4)
- **128** (avg F1: 0.7841, count: 5)
- **64** (avg F1: 0.7661, count: 7)
- **32** (avg F1: 0.6667, count: 6)
- **16** (avg F1: 0.2251, count: 3)

