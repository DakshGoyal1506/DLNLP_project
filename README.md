# GPT-2 Training Project

A flexible and configurable implementation of GPT-2 training with support for variable attention heads, YAML-based configuration, and comprehensive attention visualization.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Dataset: BabyLM](#dataset-babylm)
5. [Configuration System](#configuration-system)
6. [Variable Attention Heads](#variable-attention-heads)
7. [Attention Visualization](#attention-visualization)
8. [Performance Optimizations](#performance-optimizations)
9. [Code Documentation](#code-documentation)
10. [Bug Fixes and Known Issues](#bug-fixes-and-known-issues)
11. [Troubleshooting](#troubleshooting)
12. [Requirements](#requirements)
13. [Citation](#citation)
14. [License](#license)
15. [Acknowledgments](#acknowledgments)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Dataset

**Important**: The `babylm_data/` folder is not included in version control due to large file sizes. You need to generate it locally.

```bash
python babylm_data.py
```

**Note**: You need a Hugging Face token to access the BabyLM dataset. See [Dataset Setup](#dataset-babylm) for details.

This will create:
- `babylm_data/babylm_train.npy` (~260 MB)
- `babylm_data/babylm_val.npy` (~60 MB)
- `babylm_data/babylm_test.npy` (~8 MB)
- `babylm_data/test_examples.json` (~350 MB)

### 3. Train Model

**Windows:**
```cmd
python train_gpt2.py --config config\baseline.yaml
```

**Linux/Mac:**
```bash
python train_gpt2.py --config config/baseline.yaml
```

### 4. Visualize Attention

```bash
python train_gpt2.py --config config\test_attention.yaml
```

---

## Features

- ✅ **YAML-based Configuration**: Easy experiment management
- ✅ **Variable Attention Heads**: Different head counts per layer
- ✅ **Attention Visualization**: Generate heatmaps for all layers/heads
- ✅ **Checkpoint Management**: Automatic saving of best/final models
- ✅ **BabyLM Dataset Support**: Gated dataset integration
- ✅ **Flexible Training**: Train-only, test-only, or both modes
- ✅ **Organized Outputs**: Experiments organized by name
- ✅ **GPU/CPU Support**: Auto-detection with manual override

---

## Project Structure

```
Project/
├── config/                          # Configuration files
│   ├── baseline.yaml               # Standard GPT-2 (12 heads/layer)
│   ├── variable_heads.yaml         # Variable heads experiment
│   └── test_attention.yaml         # Attention visualization config
│
├── exp/                            # Experiment outputs
│   └── {experiment_name}/
│       ├── logs/
│       │   └── log.txt
│       ├── model_00000.pt         # Periodic checkpoints
│       ├── model_best.pt          # Best validation loss
│       └── model_final.pt         # Final checkpoint
│
├── attention_outputs/              # Attention visualizations
│   └── {experiment_name}/
│       └── {example_id}/
│           ├── layer00.png
│           ├── layer01.png
│           └── ...
│
├── babylm_data/                    # Dataset files
│   ├── babylm_train.npy
│   ├── babylm_val.npy
│   ├── babylm_test.npy
│   └── test_examples.json
│
├── train_gpt2.py                   # Main training script
├── babylm_data.py                  # Dataset preparation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Dataset: BabyLM

### Overview

The project uses the **BabyLM-community/babylm-eng** dataset, a gated dataset from Hugging Face designed for language model research with limited data.

### Dataset Splits

- **Train**: 89% of the dataset
- **Validation**: 9% of the dataset
- **Test**: 1% of the dataset

### Setup Instructions

#### Step 1: Get Hugging Face Access

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/datasets/BabyLM-community/babylm-eng
3. Click **"Access repository"** and accept the terms
4. Wait for approval (usually instant)

#### Step 2: Get Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "babylm-access")
4. Select **"Read"** access
5. Click **"Generate token"**
6. Copy the token immediately

#### Step 3: Configure Token

Open `babylm_data.py` and add your token:

```python
HF_TOKEN = "hf_YourActualTokenHere"  # Replace with your token
```

**Alternative**: Use CLI login:
```bash
huggingface-cli login
```

Then in `babylm_data.py`, set:
```python
USE_CLI_LOGIN = True
```

#### Step 4: Run Dataset Preparation

```bash
python babylm_data.py
```

This will create:
- `babylm_data/babylm_train.npy` - Training tokens
- `babylm_data/babylm_val.npy` - Validation tokens
- `babylm_data/babylm_test.npy` - Test tokens
- `babylm_data/test_examples.json` - Test examples with IDs

### Output Files

#### NumPy Files
- Tokenized sequences ready for training
- Uses GPT-2 tokenizer (tiktoken)
- Efficient binary format

#### JSON File
- `test_examples.json` contains:
  ```json
  {
    "id": 0,
    "text": "Example text...",
    "tokens": [123, 456, 789, ...]
  }
  ```

### Common Issues

**Error: "401 Unauthorized"**
- Solution: Make sure you've accepted the dataset terms and your token is valid

**Token expired**
- Solution: Generate a new token

**Whitespace in token**
- Solution: Remove any spaces before/after the token

### Alternative Datasets

If you have access issues, you can use:
- **TinyStories**: `roneneldan/TinyStories` (no auth needed)
- **WikiText**: `wikitext-103-v1` (no auth needed)
- **OpenWebText**: `openwebtext` (no auth needed)

---

## Configuration System

### Overview

The project uses YAML files for all configuration. This makes it easy to:
- Manage multiple experiments
- Reproduce results
- Share configurations
- Track experiment parameters

### Configuration File Structure

```yaml
# Experiment identification
experiment_name: "baseline"

# Model architecture
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  vocab_size: 50304
  block_size: 1024
  n_head_list: null  # For variable heads per layer

# Training hyperparameters
training:
  batch_size: 16
  sequence_length: 1024
  total_batch_size: 524288
  max_lr: 0.0006
  min_lr: 0.00006
  warmup_steps: 715
  max_steps: 19073
  weight_decay: 0.1
  grad_clip: 1.0
  eps: 1e-8

# Dataset configuration
data:
  data_dir: "./babylm_data"
  train_file: "babylm_train.npy"
  val_file: "babylm_val.npy"
  test_file: "babylm_test.npy"

# Execution control
execution:
  train: true
  test: false
  use_compile: false  # torch.compile (not supported on Windows)

# Evaluation settings
evaluation:
  val_loss_every: 250
  val_max_steps: 20
  sample_every: 250
  sample_num: 4
  sample_length: 32
  checkpoint_every: 5000
  max_checkpoints_to_keep: 5

# Output directories
output:
  exp_dir: "./exp"
  attention_dir: "./attention_outputs"

# Device configuration
device:
  device: "auto"  # auto, cuda, cpu, or mps
  dtype: "bfloat16"  # bfloat16, float32, or float16
  seed: 1337

# Checkpoint management
checkpoint:
  save_best: true
  save_final: true
  load_checkpoint: null  # Path to resume training

# Testing configuration
testing:
  checkpoint_path: "model_best.pt"  # best, final, or filename
  num_test_samples: 10
  test_sample_length: 512
  max_seq_len_viz: 50

# Logging
logging:
  log_interval: 1
  log_to_file: true
  verbose: true
```

### Available Configurations

#### baseline.yaml
- Standard GPT-2 (124M parameters)
- 12 layers, 768 embedding dimension
- Uniform 12 attention heads per layer
- Suitable for most experiments

#### variable_heads.yaml
- Experiments with different head counts per layer
- Example: `[12, 12, 12, 12, 8, 8, 8, 8, 6, 6, 6, 6]`
- Useful for efficiency research

#### test_attention.yaml
- Test-only mode
- Loads trained checkpoint
- Generates attention visualizations
- Configurable number of examples

### Creating New Configurations

1. **Copy an existing config:**
   ```bash
   copy config\baseline.yaml config\my_experiment.yaml
   ```

2. **Edit parameters:**
   - Change `experiment_name` to "my_experiment"
   - Modify model architecture, hyperparameters, etc.

3. **Run your experiment:**
   ```bash
   python train_gpt2.py --config config\my_experiment.yaml
   ```

### Training Modes

**Train Only:**
```yaml
execution:
  train: true
  test: false
```

**Test Only:**
```yaml
execution:
  train: false
  test: true
```

**Train and Test:**
```yaml
execution:
  train: true
  test: true
```

### Experiment Organization

All outputs are organized by experiment name:
- **Checkpoints**: `exp/{experiment_name}/model_*.pt`
- **Logs**: `exp/{experiment_name}/logs/log.txt`
- **Attention**: `attention_outputs/{experiment_name}/`

---

## Variable Attention Heads

### Overview

The model supports **variable numbers of attention heads per layer**, providing architectural flexibility and efficiency.

### How It Works

Instead of using the same number of heads for all layers:
- Each layer can have a different number of heads
- Head size adjusts automatically: `head_size = n_embd / n_head`
- All heads must divide evenly into `n_embd`

### Configuration

#### Uniform Heads (Default)

```yaml
model:
  n_head: 12
  n_head_list: null  # Uses n_head for all layers
```

#### Variable Heads

```yaml
model:
  n_head: 12  # Ignored when n_head_list is provided
  n_head_list: [12, 12, 12, 12, 8, 8, 8, 8, 6, 6, 6, 6]
```

### Example Patterns

#### Progressive Reduction
```yaml
n_head_list: [12, 12, 12, 10, 10, 10, 8, 8, 8, 6, 6, 6]
```
Gradually reduce heads in deeper layers.

#### Hourglass Pattern
```yaml
n_head_list: [6, 8, 10, 12, 12, 12, 12, 12, 10, 8, 6, 4]
```
More heads in middle layers, fewer at edges.

#### Efficient Late Layers
```yaml
n_head_list: [12, 12, 12, 12, 12, 12, 8, 8, 8, 6, 6, 4]
```
Full capacity early, reduce computation late.

### Constraints

1. **Divisibility**: Each `n_head` must divide `n_embd` evenly
   - For `n_embd=768`: Valid values are 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768
   - Common choices: 4, 6, 8, 12, 16, 24

2. **List Length**: `len(n_head_list)` must equal `n_layer`

3. **Head Size Calculation**:
   - 12 heads: `768 / 12 = 64` per head
   - 8 heads: `768 / 8 = 96` per head
   - 6 heads: `768 / 6 = 128` per head

### Benefits

- **Architectural Flexibility**: Experiment with different patterns
- **Efficiency**: Reduce computation in certain layers
- **Specialization**: Different representational capacities per layer
- **Research**: Investigate optimal head distribution

### Backward Compatibility

- If `n_head_list` is not provided, uses `n_head` for all layers
- Existing code works without modifications
- Old checkpoints can still be loaded

---

## Attention Visualization

### Overview

The attention visualization system extracts and plots attention weights from all layers and heads, helping you understand what your model is "paying attention to".

### Directory Structure

```
attention_outputs/
└── {experiment_name}/
    └── {example_id}/
        ├── layer00.png
        ├── layer01.png
        ├── layer02.png
        └── ...
```

### Usage

#### Step 1: Train a Model

```bash
python train_gpt2.py --config config\baseline.yaml
```

This saves checkpoints in `exp/baseline/`.

#### Step 2: Generate Visualizations

```bash
python train_gpt2.py --config config\test_attention.yaml
```

This will:
- Load the best checkpoint
- Process examples from `babylm_data/test_examples.json`
- Generate attention heatmaps for each layer
- Save plots organized by example ID

#### Step 3: View Results

Navigate to `attention_outputs/{experiment_name}/{id}/` to see plots.

### Configuration Options

Edit `config/test_attention.yaml`:

```yaml
testing:
  checkpoint_path: "model_best.pt"  # Which checkpoint
  num_test_samples: 10              # How many examples
  test_sample_length: 512           # Max tokens per example
  max_seq_len_viz: 50               # Tokens shown in labels
```

### Interpreting Plots

Each plot shows a heatmap for one layer with all attention heads:

- **X-axis**: Key tokens (what can be attended to)
- **Y-axis**: Query tokens (what is attending)
- **Color**: Attention weight (brighter = more attention)
  - Dark blue = low attention (0.0)
  - Yellow = high attention (1.0)
- **Causal mask**: Upper triangle is masked (can't attend to future)

### What to Look For

1. **Diagonal patterns**: Self-attention (token attends to itself)
2. **Vertical lines**: Multiple tokens attending to one key
3. **Horizontal bands**: One token attending to many keys
4. **Early layers**: Local patterns (nearby tokens)
5. **Later layers**: Long-range dependencies

### Technical Details

#### Attention Extraction Process

1. Load model from checkpoint
2. Forward pass through embeddings (token + position)
3. For each layer:
   - Extract Q, K, V from attention module
   - Compute attention scores: `Q @ K^T / sqrt(d_k)`
   - Apply causal mask (set future to -inf)
   - Apply softmax to get weights
   - Save weights `(n_heads, seq_len, seq_len)`

#### Visualization Format

- **Grid Layout**: 4 columns, auto-calculated rows
- **Color Scheme**: 'viridis' (dark blue → yellow)
- **Resolution**: 150 DPI
- **Format**: PNG (configurable)
- **Token Labels**: Shown if sequence ≤ 20 tokens

### Example Workflow

```bash
# Train baseline model
python train_gpt2.py --config config\baseline.yaml

# Visualize baseline attention
python train_gpt2.py --config config\test_attention.yaml

# Train variable heads model
python train_gpt2.py --config config\variable_heads.yaml

# Visualize variable heads attention
# (Edit test_attention.yaml to set experiment_name: "variable_heads")
python train_gpt2.py --config config\test_attention.yaml
```

### Tips

- **Large sequences**: Set `max_seq_len_viz` to 20-50 for readability
- **Memory**: Start with `num_test_samples: 5` if you have memory issues
- **Comparison**: Generate plots for different checkpoints to compare
- **Custom colors**: Edit `train_gpt2.py` to change `cmap` parameter

### Troubleshooting

**Cannot find test_examples.json**
- Run `babylm_data.py` to generate the file

**Plots are too crowded**
- Reduce `max_seq_len_viz` or `test_sample_length`

**Out of memory**
- Reduce `num_test_samples` or `test_sample_length`

**Matplotlib/Seaborn not installed**
- Run: `pip install matplotlib seaborn`

### Advanced Analysis

You can modify the plotting function to:
- Change color schemes
- Add specific token highlighting
- Save in different formats (PDF, SVG)
- Generate animated GIFs
- Compute attention statistics (entropy, sparsity)

---

## Performance Optimizations

### Overview

The training script includes comprehensive performance optimizations that can provide **2-5x speedup** depending on your hardware:

- ✅ TF32 acceleration on Ampere+ GPUs (A100, RTX 30/40 series)
- ✅ Flash Attention via SDPA (Linux)
- ✅ Mixed precision training (bfloat16/float16)
- ✅ Pinned memory + async GPU transfers
- ✅ torch.compile with max-autotune (Linux + CUDA)
- ✅ Fused AdamW optimizer
- ✅ GPU selection for multi-GPU servers

### Quick Start

#### Windows
```cmd
python train_gpt2.py --config config\baseline.yaml --gpu 0
```

#### Linux (Production Server)
```bash
# Set environment variables for maximum performance
export PYTORCH_SDP_KERNEL=flash
export NVIDIA_TF32_OVERRIDE=1

# Train with all optimizations
python train_gpt2.py --config config/baseline.yaml --gpu 0
```

### Optimizations Explained

#### 1. TF32 Acceleration (Ampere+ GPUs)
- **Speedup**: ~1.5-2x on A100/RTX 30/40 series
- **Enabled**: Automatically on CUDA devices
- Uses TensorFloat-32 format for matmuls with negligible accuracy loss

#### 2. Flash Attention (SDPA)
- **Speedup**: ~2-4x faster attention, ~50% less memory
- **Enabled**: Linux + CUDA with `PYTORCH_SDP_KERNEL=flash`
- Uses memory-efficient Flash Attention kernel

#### 3. Mixed Precision Training
- **Speedup**: ~2x faster with ~50% less memory
- **Device-specific**:
  - CUDA (Ampere+): `bfloat16` (default, better stability)
  - CUDA (older): `float16`
  - MPS (Apple): `float16` (automatic)

```yaml
device:
  dtype: "bfloat16"  # or "float16" or "float32"
```

#### 4. Pinned Memory + Async GPU Transfer
- **Speedup**: ~10-30% reduction in data loading overhead
- **Enabled**: Automatically on CUDA devices
- Uses page-locked host memory for faster H2D transfers

#### 5. torch.compile (Linux + CUDA Only)
- **Speedup**: ~1.3-1.8x on top of other optimizations
- **Requirements**: Linux + CUDA
- JIT compiles model into optimized CUDA kernels

```yaml
execution:
  use_compile: true  # Only works on Linux + CUDA
```

#### 6. Fused AdamW
- **Speedup**: ~10-20% faster optimizer step
- **Enabled**: Automatically on CUDA when available

#### 7. GPU Selection
```bash
# Method 1 - Command line (recommended)
python train_gpt2.py --config config/baseline.yaml --gpu 0

# Method 2 - Environment variable
CUDA_VISIBLE_DEVICES=2 python train_gpt2.py --config config/baseline.yaml
```

### Expected Speedups

| Configuration | Relative Speed | Notes |
|---------------|---------------|-------|
| Baseline (fp32, no optimizations) | 1.0x | - |
| + TF32 | 1.5x | Ampere+ only |
| + Mixed precision (bf16) | 2.0x | Cumulative |
| + Flash Attention | 3.0x | Cumulative |
| + Pinned memory | 3.3x | Cumulative |
| + torch.compile | 4.5x | Linux + CUDA |
| **All optimizations** | **4-5x** | Best case |

### Performance Tips

#### 1. Reduce Validation Overhead
```yaml
evaluation:
  val_interval: 1000  # Was 250
  val_loss_steps: 5   # Reduce mini-evals
```

#### 2. Batch Size Tuning
- **Larger B**: Better GPU utilization, needs more memory
- **Keep total_batch_size constant**: Let grad_accum_steps adjust automatically

```yaml
training:
  batch_size: 32  # Increase if you have memory
  total_batch_size: 524288  # Keep constant
```

#### 3. A6000 Training (Linux)
```bash
# Single command for maximum speed on A6000
export PYTORCH_SDP_KERNEL=flash && export NVIDIA_TF32_OVERRIDE=1 && python train_gpt2.py --config config/training.yaml --gpu 0
```

**Expected Results**:
- **Tokens/sec**: ~60-80K on A6000
- **Training time**: ~5-6 hours for full run (19,073 steps)
- **Speedup**: ~5-6x faster than baseline

---

## Code Documentation

### Documentation Style

The codebase follows **Google-style docstrings** with comprehensive dimension annotations throughout.

#### Example Class Documentation
```python
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention.
    
    Uses PyTorch's scaled_dot_product_attention for efficient computation with
    automatic Flash Attention kernel selection when available.
    
    Attributes:
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        head_size: Dimension per attention head (n_embd // n_head).
    """
    
    def forward(self, x):
        """Compute multi-head causal self-attention.
        
        Args:
            x: Input tensor of shape (B, T, C).
            
        Returns:
            Attention output tensor of shape (B, T, C).
        """
        B, T, C = x.size()  # (B, T, C)
        # ...existing code...
```

#### Dimension Annotations
Consistent dimension comments throughout:
- `(B, T, C)` for input/output tensors
- `(B, n_head, T, head_size)` for attention tensors
- `(B, T, vocab_size)` for logits
- `(T,)` for position indices
- `(B, T, 4*C)` for MLP hidden layer

#### Documented Components

**Classes**:
- CausalSelfAttention - Multi-head attention with Flash Attention
- MLP - Feed-forward network with GELU
- Block - Transformer block with pre-normalization
- GPTConfig - Model configuration dataclass
- GPT - Main GPT model
- DataLoaderLiteConfig - Optimized data loader with pinned memory

**Functions**:
- `load_tokens()` - Load tokenized data from numpy
- `load_config()` - YAML configuration loading
- `setup_experiment_dirs()` - Directory structure creation
- `get_lr()` - Learning rate schedule (warmup + cosine decay)
- `extract_attention_patterns()` - Extract attention weights
- `plot_layer_attention()` - Visualize attention heatmaps

**Training/Testing Loops**:
- Main training loop with validation, checkpointing, sample generation
- Gradient accumulation with mixed precision
- Testing loop with attention visualization

---

## Bug Fixes and Known Issues

### Fixed Issues

#### 1. Checkpoint Tracker IndexError
- **Fixed**: Empty checkpoint list check before accessing
- **Impact**: Prevents crash on short training runs

#### 2. Division by Zero in Test Evaluation
- **Fixed**: Check for empty test batches before computing metrics
- **Impact**: Handles small datasets gracefully

#### 3. Configuration Parameter Hardcoding
- **Fixed**: All parameters now use YAML config values
- **Impact**: Complete configurability

### Linting Notes

Some static analyzer warnings are false positives:
- Conditional variable definitions (model, optimizer defined within `if train_flag`)
- Module dictionary access patterns
- torch.version.cuda attribute (exists at runtime)

These don't affect runtime execution.

---

## Troubleshooting

### Dataset Issues

**401 Unauthorized Error**
- Ensure you've accepted the BabyLM dataset terms
- Check your Hugging Face token is valid
- Regenerate token if needed
- Remove any whitespace from token

**Token expired**
- Go to https://huggingface.co/settings/tokens
- Generate a new token
- Update `babylm_data.py`

**Cannot find dataset**
- Make sure you ran `babylm_data.py`
- Check `babylm_data/` directory exists
- Verify NumPy files were created

### Training Issues

**Out of memory**
- Reduce `batch_size` in config
- Reduce `sequence_length`
- Use gradient accumulation (increase `total_batch_size` relative to `batch_size`)
- For RTX 3080 Laptop (16GB): Use `batch_size: 8`

**Slow training**
- Use GPU if available
- Enable mixed precision: `dtype: "bfloat16"`
- On Linux: Enable torch.compile
- Set environment variables: `PYTORCH_SDP_KERNEL=flash` and `NVIDIA_TF32_OVERRIDE=1`

**Slow training on Windows**
- Use WSL2 + Ubuntu for 2-3x speedup
- torch.compile not supported on Windows
- Flash Attention limited on Windows

**Loss is NaN**
- Reduce learning rate
- Check `grad_clip` is enabled
- Verify data preprocessing

### Attention Visualization Issues

**Cannot find test_examples.json**
- Run `babylm_data.py` first
- Check file exists in `babylm_data/`

**Plots too crowded**
- Reduce `max_seq_len_viz` (try 20-30)
- Reduce `test_sample_length`

**Out of memory during visualization**
- Reduce `num_test_samples`
- Reduce `test_sample_length`

**Matplotlib errors**
- Install: `pip install matplotlib seaborn`
- Update: `pip install --upgrade matplotlib seaborn`

### Configuration Issues

**YAML parsing error**
- Check indentation (use spaces, not tabs)
- Avoid scientific notation (use 0.0006 instead of 6e-4)
- Validate YAML syntax online

**Cannot find config file**
- Check path is correct
- Use absolute path if needed
- Verify file has `.yaml` extension

**Missing configuration key**
- Compare with `baseline.yaml`
- Ensure all required sections present
- Check for typos in key names

### Optimization Issues

**torch.compile fails**
- Expected on Windows - automatically disabled
- Requires Linux + CUDA
- Check PyTorch version >= 2.0

**Flash Attention not available**
- Check PyTorch version >= 2.0
- Set environment variable: `PYTORCH_SDP_KERNEL=flash`
- Verify: `torch.backends.cuda.flash_sdp_enabled()`

**Wrong GPU being used**
- Use `--gpu` flag: `python train_gpt2.py --config config/baseline.yaml --gpu 2`
- Or set `CUDA_VISIBLE_DEVICES` environment variable
- Check with: `nvidia-smi`

### General Issues

**Checkpoint not loading**
- Verify checkpoint file exists
- Check `checkpoint_path` in config
- Use "model_best.pt" or "model_final.pt"

**Wrong device**
- Set `device: "cuda"` or `device: "cpu"` explicitly
- Check CUDA availability: `torch.cuda.is_available()`

**Import errors**
- Install all requirements: `pip install -r requirements.txt`
- Check Python version (3.8+)

---

## Requirements

### Python Packages

```
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.5.0
datasets>=2.14.0
huggingface-hub>=0.17.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA (optional but recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 5GB for dataset, 10GB for experiments

---

## Citation

If you use this code for research, please cite:

```bibtex
@misc{gpt2-babylm-training,
  author = {Your Name},
  title = {GPT-2 Training with Variable Attention Heads and Visualization},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/your-repo}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **BabyLM Challenge**: For providing the dataset
- **Hugging Face**: For datasets and tokenizers infrastructure
- **Andrej Karpathy**: For GPT-2 implementation inspiration
- **OpenAI**: For the original GPT-2 architecture

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

---

## Changelog

### Version 1.2 (Current - November 2025)
- ✅ Complete Google-style documentation with dimension annotations
- ✅ Performance optimizations (TF32, Flash Attention, torch.compile)
- ✅ GPU selection via command-line flag
- ✅ Pinned memory and async GPU transfers
- ✅ Fused AdamW optimizer
- ✅ Mixed precision training (bfloat16/float16)
- ✅ Bug fixes for edge cases (empty checkpoints, small datasets)
- ✅ Training and testing loop documentation

### Version 1.1
- ✅ All configuration parameters from YAML (no hardcoded values)
- ✅ Gradient clipping from config
- ✅ Sample generation parameters from config
- ✅ Top-k sampling parameter support

### Version 1.0
- ✅ YAML-based configuration system
- ✅ Variable attention heads per layer
- ✅ Attention visualization system
- ✅ BabyLM dataset integration
- ✅ Basic documentation

### Planned Features
- [ ] Multi-GPU DDP training support
- [ ] Gradient checkpointing toggle
- [ ] Interactive attention visualizations (web interface)
- [ ] Attention statistics and analysis (entropy, sparsity)
- [ ] WandB/TensorBoard integration
- [ ] Automatic mixed precision (AMP) scaler
