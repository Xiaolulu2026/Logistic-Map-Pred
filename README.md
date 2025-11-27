# Logistic-Map-Pred

Predicting the bifurcation behavior of chaotic systems (Logistic Map) using neural networks (GRU-MLP). Supports predicting chaotic regions from ordered regions, or conversely predicting ordered regions from chaotic regions.

```bash
# Basic environment (required)
pip install numpy matplotlib scikit-learn tqdm

# GRU model support (optional)
pip install torch
```

**Python Version**: 3.8+

### Basic Execution

```bash
# 1. Simplest run (using default parameters)
python gru_mlp_bifurcation.py

# 2. Time series prediction with GRU model
python gru_mlp_bifurcation.py --mode ts --model gru

# 3. Predict sparse regions (ordered) from dense regions (chaotic)
python gru_mlp_bifurcation.py --mode ts --model gru --train_side dense

# 4. View all parameters
python gru_mlp_bifurcation.py --help
```

## Detailed Explanation of Running Modes

### Mode 1: Map Mode (Mapping Learning)

Learn the mapping function f: (r, x) → x' to predict trajectories through iteration

```bash
python gru_mlp_bifurcation.py \
  --mode map \
  --model mlp \
  --hidden 256,256 \
  --train_side sparse \
  --max_train_iter 1200
```

**Principle**:
- Training data: Extract (r, x_t) → x_{t+1} pairs from trajectories
- Prediction: Start from random initial values and iteratively apply the learned mapping
- Application scenario: Cases requiring long-term simulation


### Mode 2: TS Mode (Time Series Prediction)

Predict future values using sliding windows, supporting GRU models

```bash
python gru_mlp_bifurcation.py \
  --mode ts \
  --model gru \
  --ts_window 16 \
  --ts_horizon 16 \
  --ts_strategy iter \
  --gru_hidden_size 256 \
  --gru_num_layers 2 \
  --use_curriculum
```

**Principle**:
- Training data: Sliding window (r, [x_{t-w}, ..., x_t]) → [x_{t+1}, ..., x_{t+h}]
- Two strategies:
  - `direct`: Directly predict h future steps (multi-output)
  - `iter`: Iterative prediction, predicting 1 step each time (recommended)
- GRU advantage: Captures temporal dependencies, suitable for chaotic systems

## Core Parameter Explanation

### Basic Parameters

```bash
--r_min 2.5              # Minimum value of r parameter range
--r_max 4.0              # Maximum value of r parameter range
--num_r 400              # Number of r sampling points
--split_r 3.56994        # Boundary between sparse/dense regions
--seed 42                # Random seed
--num_iterations 3000    # Number of iterations per r
--num_discard 1000       # Number of transient steps to discard
```

### Model Parameters

```bash
--model {mlp,ridge,svr,gru}  # Model type
--hidden 256,256             # MLP hidden layer structure
--max_train_iter 1200        # Maximum training epochs
--lr 0.001                   # Learning rate
```

### Training Direction

```bash
--train_side sparse      # Predict dense regions (chaotic) from sparse regions (ordered)
--train_side dense       # Predict sparse regions (ordered) from dense regions (chaotic)
```

### Time Series Parameters (only mode=ts)

```bash
--ts_window 16           # Input window length
--ts_horizon 16          # Prediction horizon
--ts_strategy iter       # Prediction strategy: iter (iterative) or direct (direct)
--ts_max_windows_per_r 800  # Maximum number of windows per r
```

### GRU Model Parameters

```bash
--gru_hidden_size 256    # GRU hidden layer size
--gru_num_layers 2       # Number of GRU layers
--gru_batch_size 256     # Batch size
--gru_dropout 0.1        # Dropout ratio
```

### Physical Priors and Regularization

```bash
# Residual connection (recommended)
--gru_residual           # GRU output as residual to Logistic Map baseline
--gru_residual_lambda 0.001  # Residual regularization coefficient

# Rollout consistency (improve long-term stability)
--rollout_consistency_k 4         # Rollout steps
--rollout_consistency_lambda 0.001  # Consistency loss weight

# Input perturbation (data augmentation)
--ts_input_jitter 0.01   # Additive Gaussian noise to windows during training

# r feature engineering
--gru_r_poly_degree 2    # Polynomial feature degree of r
--gru_r_sin_cos          # Add sin/cos trigonometric features
--r_norm_to_unit         # Normalize r to [-1,1]

# Gradient clipping
--grad_clip 1.0          # Gradient clipping threshold
```

### Curriculum Learning

```bash
--use_curriculum         # Enable curriculum learning
--curriculum_stages 3    # Number of stages from simple to complex
```

### Advanced Features

```bash
# Automatic detection of chaotic boundary
--auto_split             # Automatically detect split_r based on Lyapunov exponent
--auto_split_eps 1e-4    # Positive value threshold
--auto_split_smooth 5    # Smoothing window

# Data augmentation
--ts_phys_aug_sparse     # Add physically synthesized data near boundaries
--ts_phys_aug_mode auto  # auto: around split_r automatically; manual: manual range
--ts_phys_band 0.15      # Augmentation bandwidth in auto mode

# Neuron pruning
--prune_after_train      # Prune low-importance neurons after training
--prune_threshold 0.09   # Pruning threshold
--prune_max_neurons 4    # Maximum number of neurons to prune

# Additional evaluation metrics
--eval_extra             # Calculate SSIM, EMD, Period Consistency F1
--per_r_csv              # Export per-r detailed metrics CSV
```


## Output File Description

### Visualization Images

| Filename | Description |
|----------|-------------|
| `bifurcation_scatter.png` | True bifurcation diagram (scatter) |
| `bifurcation_scatter_pred.png` | Predicted bifurcation diagram comparison |
| `lyapunov_true.png` | Lyapunov exponent curve |

### Data Files

| Filename | Description | Trigger Parameter |
|----------|-------------|-------------------|
| `metrics_extra.csv` | Extended evaluation metrics | `--eval_extra` |
| `per_r_metrics.csv` | Per-r detailed metrics and bin statistics | `--per_r_csv` |


---
