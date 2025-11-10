# Training Configuration

Single optimized configuration file for GPT-2 training on A6000 GPU (Linux server).

## Configuration File

### `training.yaml` - Production Training (A6000 Optimized)

**Optimized for maximum training speed on A6000 (48GB VRAM) running Linux.**

**Key Features:**
- ✅ All performance optimizations enabled (TF32, Flash Attention, torch.compile)
- ✅ Batch size: 32 (optimized for A6000)
- ✅ bfloat16 mixed precision
- ✅ Reduced validation overhead (every 1000 steps, 5 batches)
- ✅ Sample generation disabled during training for max speed
- ✅ Uniform 12 attention heads per layer: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

**Expected Performance:**
- Speedup: ~5-6x over baseline
- Tokens/sec: ~60-80K on A6000
- Training time: ~5-6 hours for 19K steps

## Usage

### Training on Linux Server

```bash
# Set environment variables for maximum performance
export PYTORCH_SDP_KERNEL=flash
export NVIDIA_TF32_OVERRIDE=1

# Train with GPU selection (if multiple GPUs)
python train_gpt2.py --config config/training.yaml --gpu 0

# Or let it auto-select GPU
python train_gpt2.py --config config/training.yaml
```

### Testing Mode

To test a trained model, edit `training.yaml`:

```yaml
execution:
  train: false
  test: true
```

Then run:
```bash
python train_gpt2.py --config config/training.yaml
```

## Configuration Sections

### Model Architecture
```yaml
model:
  n_layer: 12
  n_embd: 768
  n_head: 12
  n_head_list: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
  block_size: 1024
  vocab_size: 50304
```

### Optimization Settings
```yaml
device:
  dtype: "bfloat16"                # Best for A6000
  enable_tf32: true                # TF32 acceleration
  enable_cudnn_benchmark: true

execution:
  use_compile: true                # torch.compile for 2x speedup
  compile_mode: "max-autotune"
```

### Training Settings (Optimized for Speed)
```yaml
training:
  batch_size: 32                   # A6000 optimized
  
evaluation:
  val_interval: 1000               # Validate less frequently
  val_steps: 5                     # Fewer validation batches
  generate_samples: false          # Disabled for max speed
```

## Customization

### Adjust for Different Memory
If you get OOM errors:
```yaml
training:
  batch_size: 16                   # Reduce from 32
```

### More Frequent Validation
If you want to monitor training more closely:
```yaml
evaluation:
  val_interval: 500                # Validate more often
  val_steps: 10                    # More validation batches
```

### Enable Sample Generation
To see text samples during training:
```yaml
evaluation:
  generate_samples: true
  sample_interval: 1000
```

## Performance Tips

1. **Always set environment variables before training:**
   ```bash
   export PYTORCH_SDP_KERNEL=flash
   export NVIDIA_TF32_OVERRIDE=1
   ```

2. **Select specific GPU on multi-GPU server:**
   ```bash
   python train_gpt2.py --config config/training.yaml --gpu 0
   ```

3. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Check training is using optimizations:**
   - Look for "✓ Enabled TF32" in output
   - Look for "✓ Model compiled successfully!" (Linux only)
   - Look for "loaded ... tokens (pinned memory)"

## Additional Documentation

- **Main Documentation**: `../README.md` - Complete project documentation including performance optimizations, troubleshooting, and configuration details
