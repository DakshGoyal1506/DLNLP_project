import os
import math
import time
import inspect
import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention.
    
    Uses PyTorch's scaled_dot_product_attention for efficient computation with
    automatic Flash Attention kernel selection when available.
    
    Attributes:
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        head_size: Dimension per attention head (n_embd // n_head).
        c_attn: Linear layer for computing Q, K, V projections.
        c_proj: Output projection layer.
    """

    def __init__(self, config, n_head):
        """Initialize causal self-attention layer.
        
        Args:
            config: Model configuration object with n_embd attribute.
            n_head: Number of attention heads for this layer.
            
        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        super().__init__()
        assert config.n_embd % n_head == 0, f"n_embd ({config.n_embd}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """Compute multi-head causal self-attention.
        
        Args:
            x: Input tensor of shape (B, T, C).
            
        Returns:
            Attention output tensor of shape (B, T, C).
        """
        B, T, C = x.size()  # (B, T, C)
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each: (B, T, C)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)  # (B, T, C)
        return y

class MLP(nn.Module):
    """Feed-forward network with GELU activation.
    
    Implements the MLP block: Linear -> GELU -> Linear with 4x hidden dimension.
    
    Attributes:
        c_fc: First linear layer that expands dimension by 4x.
        gelu: GELU activation function with tanh approximation.
        c_proj: Second linear layer that projects back to original dimension.
    """

    def __init__(self, config):
        """Initialize MLP block.
        
        Args:
            config: Model configuration object with n_embd attribute.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (B, T, C).
            
        Returns:
            Output tensor of shape (B, T, C).
        """
        x = self.c_fc(x)      # (B, T, 4*C)
        x = self.gelu(x)      # (B, T, 4*C)
        x = self.c_proj(x)    # (B, T, C)
        return x

class Block(nn.Module):
    """Transformer block with pre-normalization.
    
    Implements: x = x + attn(ln(x)) followed by x = x + mlp(ln(x)).
    
    Attributes:
        ln_1: Layer normalization before attention.
        attn: Multi-head causal self-attention layer.
        ln_2: Layer normalization before MLP.
        mlp: Feed-forward network.
    """

    def __init__(self, config, n_head):
        """Initialize transformer block.
        
        Args:
            config: Model configuration object.
            n_head: Number of attention heads for this block.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, n_head)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, C).
            
        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))  # (B, T, C)
        x = x + self.mlp(self.ln_2(x))   # (B, T, C)
        return x

@dataclass
class GPTConfig:
    """Configuration for GPT model architecture.
    
    Attributes:
        block_size: Maximum sequence length (context window).
        vocab_size: Size of vocabulary (50257 for GPT-2).
        n_layer: Number of transformer layers.
        n_head: Default number of attention heads per layer.
        n_embd: Embedding dimension.
        n_head_list: Optional list specifying heads per layer. If None, uses n_head for all layers.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_head_list: list = None

class GPT(nn.Module):
    """GPT language model with configurable architecture.
    
    Implements a decoder-only transformer with token and position embeddings,
    multiple transformer blocks, and a language modeling head.
    
    Attributes:
        config: Model configuration.
        transformer: ModuleDict containing:
            - wte: Token embedding layer.
            - wpe: Position embedding layer.
            - h: ModuleList of transformer blocks.
            - ln_f: Final layer normalization.
        lm_head: Linear layer for predicting next token logits.
    """

    def __init__(self, config):
        """Initialize GPT model.
        
        Args:
            config: GPTConfig object specifying model architecture.
            
        Raises:
            AssertionError: If n_head_list length doesn't match n_layer.
        """
        super().__init__()
        self.config = config

        if config.n_head_list is None:
            n_head_list = [config.n_head] * config.n_layer
        else:
            n_head_list = config.n_head_list
            assert len(n_head_list) == config.n_layer, f"n_head_list length ({len(n_head_list)}) must match n_layer ({config.n_layer})"

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, n_head=n_head_list[i]) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for linear and embedding layers.
        
        Uses scaled initialization for residual projection layers marked with
        NANOGPT_SCALE_INIT attribute.
        
        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        """Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (B, T).
            target: Optional target indices of shape (B, T) for computing loss.
            
        Returns:
            Tuple of (logits, loss) where:
                - logits: Token predictions of shape (B, T, vocab_size).
                - loss: Cross-entropy loss (float) if target provided, else None.
                
        Raises:
            AssertionError: If sequence length T exceeds block_size.
        """
        B, T = idx.size()  # (B, T)
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer['wpe'](pos)  # (T, C)
        tok_emb = self.transformer['wte'](idx)  # (B, T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        for block in self.transformer.h:
            x = block(x)  # (B, T, C)

        x = self.transformer['ln_f'](x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), eps=1e-8, fused_from_config=True):
        """Create AdamW optimizer with weight decay only on 2D parameters.
        
        Separates parameters into two groups:
        - decay_params: 2D+ tensors (weights) that get weight decay.
        - nodecay_params: 1D tensors (biases, layer norms) without weight decay.
        
        Args:
            weight_decay: L2 regularization coefficient for weight matrices.
            learning_rate: Initial learning rate.
            device_type: Device type ("cuda", "cpu", "mps").
            betas: Adam beta parameters (momentum, RMSprop).
            eps: Adam epsilon for numerical stability.
            fused_from_config: Whether to use fused AdamW kernel if available.
            
        Returns:
            Configured AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda" and fused_from_config
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    """Load tokenized data from numpy file.
    
    Args:
        filename: Path to .npy file containing token indices.
        
    Returns:
        LongTensor of token indices.
    """
    tokens = np.load(filename).astype(np.int32)
    return torch.tensor(tokens, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val', 'test'}

        # load the data file from babylm_data directory
        data_root = "babylm_data"
        data_file = os.path.join(data_root, f"babylm_{split}.npy")
        assert os.path.exists(data_file), f"data file not found: {data_file}"
        print(f"loading data from {data_file}")
        self.tokens = load_tokens(data_file)
        print(f"loaded {len(self.tokens)} tokens")
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # wrap BEFORE slicing to guarantee length B*T+1
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def setup_experiment_dirs(cfg):
    """Create experiment-specific directory structure.
    
    Creates directories for experiment outputs, logs, and attention visualizations.
    
    Args:
        cfg: Configuration dictionary with experiment_name and output paths.
        
    Returns:
        Dictionary with paths:
            - exp_dir: Main experiment directory.
            - attention_dir: Attention visualization output directory.
            - log_dir: Training logs directory.
    """
    exp_name = cfg['experiment_name']
    
    exp_dir = Path(cfg['output']['exp_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    attention_dir = Path(cfg['output']['attention_dir']) / exp_name
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'exp_dir': exp_dir,
        'attention_dir': attention_dir,
        'log_dir': log_dir
    }

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GPT-2 model with configurable parameters')
parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use (e.g., 0, 1, 2). If not specified, uses config or auto-detection.')
args = parser.parse_args()

# Load configuration
cfg = load_config(args.config)
print(f"Loaded configuration from: {args.config}")
print(f"Experiment name: {cfg['experiment_name']}")

# Setup experiment directories
dirs = setup_experiment_dirs(cfg)

# -----------------------------------------------------------------------------
# Simple launch:
# python train_gpt2.py --config config/baseline.yaml
# Multi-GPU: python train_gpt2.py --config config/baseline.yaml --gpu 0

# Detect platform
import platform
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Device setup with GPU selection
device_cfg = cfg['device']['device']

# Handle GPU selection from command line or config
if args.gpu is not None:
    # Command-line GPU specification takes precedence
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = "cuda"
    print(f"Using GPU {args.gpu} (via command-line argument)")
elif device_cfg == "auto":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
else:
    device = device_cfg

print(f"Platform: {platform.system()}")
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

# Set random seed
seed = cfg['device']['seed']
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Training hyperparameters from config
B = cfg['training']['batch_size']
T = cfg['training']['sequence_length']
total_batch_size = cfg['training']['total_batch_size']
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"calculated gradient accumulation steps: {grad_accum_steps}")

# Extract execution flags
train_flag = cfg['execution']['train']
test_flag = cfg['execution']['test']

# Update DataLoaderLite to use config paths
class DataLoaderLiteConfig:
    """Optimized data loader with pinned memory for faster GPU transfer.
    
    Loads tokenized data and provides batches for language modeling.
    Uses pinned memory on CUDA devices for faster host-to-device transfers.
    
    Attributes:
        B: Batch size.
        T: Sequence length.
        pin_memory: Whether to use pinned memory.
        tokens: Full dataset as LongTensor.
        current_position: Current position in dataset.
    """
    
    def __init__(self, B, T, file_path, pin_memory=True):
        """Initialize data loader.
        
        Args:
            B: Batch size.
            T: Sequence length (context window).
            file_path: Path to .npy file containing tokenized data.
            pin_memory: Use pinned memory for faster GPU transfer (CUDA only).
            
        Raises:
            AssertionError: If file_path doesn't exist.
        """
        self.B = B
        self.T = T
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        assert os.path.exists(file_path), f"data file not found: {file_path}"
        print(f"loading data from {file_path}")
        tokens = load_tokens(file_path)
        
        if self.pin_memory:
            self.tokens = tokens.pin_memory()
            print(f"loaded {len(self.tokens)} tokens (pinned memory)")
        else:
            self.tokens = tokens
            print(f"loaded {len(self.tokens)} tokens")
            
        self.current_position = 0

    def reset(self):
        """Reset dataset position to beginning."""
        self.current_position = 0

    def next_batch(self):
        """Get next batch of data with autoregressive targets.
        
        Returns:
            Tuple of (x, y) where:
                - x: Input tokens of shape (B, T).
                - y: Target tokens of shape (B, T), shifted by 1.
        """
        B, T = self.B, self.T
        
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
            
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        if self.pin_memory:
            x = (buf[:-1]).view(B, T).pin_memory()  # (B, T)
            y = (buf[1:]).view(B, T).pin_memory()   # (B, T)
        else:
            x = (buf[:-1]).view(B, T)  # (B, T)
            y = (buf[1:]).view(B, T)   # (B, T)
            
        self.current_position += B * T
        return x, y

# Load data based on execution flags
data_dir = Path(cfg['data']['data_dir'])
train_file = data_dir / cfg['data']['train_file']
val_file = data_dir / cfg['data']['val_file']
test_file = data_dir / cfg['data']['test_file']

if train_flag:
    print("=" * 80)
    print("TRAINING MODE ENABLED")
    print("=" * 80)
    # Enable pinned memory for CUDA devices
    use_pin_memory = (device_type == "cuda")
    train_loader = DataLoaderLiteConfig(B=B, T=T, file_path=str(train_file), pin_memory=use_pin_memory)
    val_loader = DataLoaderLiteConfig(B=B, T=T, file_path=str(val_file), pin_memory=use_pin_memory)
    test_loader = DataLoaderLiteConfig(B=B, T=T, file_path=str(test_file), pin_memory=use_pin_memory)

if test_flag:
    print("=" * 80)
    print("TESTING MODE ENABLED")
    print("=" * 80)
    if not train_flag:  # Only load test data if not already loaded
        use_pin_memory = (device_type == "cuda")
        test_loader = DataLoaderLiteConfig(B=B, T=T, file_path=str(test_file), pin_memory=use_pin_memory)

# Get learning rate parameters from config
max_lr = cfg['training']['max_lr']
min_lr = cfg['training']['min_lr']
warmup_steps = cfg['training']['warmup_steps']
max_steps = cfg['training']['max_steps']

# Get checkpoint management parameters
MAX_CHECKPOINTS_TO_KEEP = cfg['evaluation']['max_checkpoints_to_keep']
save_best = cfg['checkpoint']['save_best']
save_final = cfg['checkpoint']['save_final']

# Create model
if train_flag:
    # Create model from config
    model_config = GPTConfig(**cfg['model'])
    model = GPT(model_config)
    
    # model = GPT.from_pretrained("gpt2")  # Or init from OpenAI GPT-2
    model.to(device)
    
    # PyTorch 2.0 compile for faster training (Linux + CUDA only)
    use_compile = cfg['execution'].get('use_compile', False)
    can_compile = (device_type == "cuda" and IS_LINUX)
    
    if use_compile and can_compile:
        print("Compiling model with torch.compile(mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")
        print("✓ Model compiled successfully!")
    elif use_compile and not can_compile:
        print(f"⚠ torch.compile requested but not available (device: {device_type}, platform: {platform.system()})")
        print("  torch.compile requires Linux + CUDA. Proceeding without compilation.")
        use_compile = False
    else:
        use_compile = False

if test_flag:
    # Load checkpoint and recreate model
    checkpoint_path = str(dirs['exp_dir'] / cfg['testing']['checkpoint_path'])
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"Loaded model from step {checkpoint['step']}")
    print(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Checkpoint test_loss: {checkpoint.get('test_loss', 'N/A')}")

def get_lr(it):
    """Learning rate schedule with warmup and cosine decay.
    
    Implements three phases:
    1. Linear warmup from 0 to max_lr.
    2. Cosine decay from max_lr to min_lr.
    3. Constant min_lr after max_steps.
    
    Args:
        it: Current training step.
        
    Returns:
        Learning rate for current step.
    """
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Training mode setup
if train_flag:
    optimizer = model.configure_optimizers(
        weight_decay=cfg['training']['weight_decay'], 
        learning_rate=max_lr, 
        device_type=device_type,
        betas=tuple(cfg['training']['betas']),
        eps=cfg['training']['eps'],
        fused_from_config=cfg['training']['fused']
    )

    log_dir = str(dirs['log_dir'])
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass
    
    # Checkpoint management: track best checkpoints
    checkpoint_tracker = []  # List of tuples: (val_loss, checkpoint_path, step)
    
    # Cache evaluation intervals to avoid repeated dict lookups
    val_interval = cfg['evaluation']['val_interval']
    sample_interval = cfg['evaluation']['sample_interval']
    checkpoint_interval = cfg['evaluation']['checkpoint_interval']

# =============================================================================
# TRAINING LOOP
# =============================================================================
if train_flag:
    """Main training loop with validation, checkpointing, and sample generation.
    
    For each training step:
    1. Periodically evaluate on validation and test sets
    2. Save checkpoints and manage top-k best models
    3. Optionally generate text samples
    4. Perform forward pass with gradient accumulation
    5. Update model parameters with learning rate schedule
    6. Log metrics (loss, learning rate, throughput)
    """
    
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # =====================================================================
        # VALIDATION AND TEST EVALUATION
        # =====================================================================
        if step % val_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    # Async GPU transfer with non_blocking for pinned memory
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    # Use dtype from config (bfloat16 on Ampere+, float16 on MPS/older GPUs)
                    dtype_str = cfg['device'].get('dtype', 'bfloat16')
                    if device_type == "mps":
                        dtype = torch.float16  # MPS doesn't support bfloat16 well
                    elif dtype_str == "bfloat16":
                        dtype = torch.bfloat16
                    elif dtype_str == "float16":
                        dtype = torch.float16
                    else:
                        dtype = torch.float32
                        
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
            # Evaluate test loss
            test_loader.reset()
            with torch.no_grad():
                test_loss_accum = 0.0
                test_loss_steps = 20
                for _ in range(test_loss_steps):
                    x, y = test_loader.next_batch()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(x, y)
                    loss = loss / test_loss_steps
                    test_loss_accum += loss.detach()
            print(f"test loss: {test_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} test {test_loss_accum.item():.4f}\n")
            
            # =================================================================
            # CHECKPOINT MANAGEMENT
            # =================================================================
            # Save checkpoints and manage top-k best models by validation loss.
            # Maintains only the best MAX_CHECKPOINTS_TO_KEEP checkpoints,
            # plus optional best and final checkpoints.
            if step > 0 and (step % checkpoint_interval == 0 or last_step):
                current_val_loss = val_loss_accum.item()
                current_test_loss = test_loss_accum.item()
                checkpoint_path = os.path.join(str(dirs['exp_dir']), f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': current_val_loss,
                    'test_loss': current_test_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Track this checkpoint
                checkpoint_tracker.append((current_val_loss, checkpoint_path, step))
                
                # Keep only top MAX_CHECKPOINTS_TO_KEEP checkpoints based on validation loss
                checkpoint_tracker.sort(key=lambda x: x[0])  # Sort by val_loss (lower is better)
                
                # Remove checkpoints beyond top-k
                if len(checkpoint_tracker) > MAX_CHECKPOINTS_TO_KEEP:
                    for _, old_checkpoint_path, _ in checkpoint_tracker[MAX_CHECKPOINTS_TO_KEEP:]:
                        if os.path.exists(old_checkpoint_path):
                            os.remove(old_checkpoint_path)
                            print(f"Removed checkpoint: {old_checkpoint_path}")
                    checkpoint_tracker = checkpoint_tracker[:MAX_CHECKPOINTS_TO_KEEP]
                
                # Save the best checkpoint separately
                if save_best and checkpoint_tracker and checkpoint_tracker[0][2] == step:  # If current is the best
                    best_checkpoint_path = os.path.join(str(dirs['exp_dir']), "model_best.pt")
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"New best model! Saved to {best_checkpoint_path}")
                
                # Always save the final checkpoint
                if save_final and last_step:
                    final_checkpoint_path = os.path.join(str(dirs['exp_dir']), "model_final.pt")
                    torch.save(checkpoint, final_checkpoint_path)
                    print(f"Saved final checkpoint to {final_checkpoint_path}")

        # # evaluate hellaswag (commented out - not needed for babylm dataset)
        # if (step % 250 == 0 or last_step) and (not use_compile):
        #     num_correct_norm = 0
        #     num_total = 0
        #     for example in iterate_examples("val"):
        #         _, tokens, mask, label = render_example(example)
        #         tokens = tokens.to(device)
        #         mask = mask.to(device)
        #         with torch.no_grad():
        #             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #                 logits, loss = model(tokens)
        #             pred_norm = get_most_likely_row(tokens, mask, logits)
        #         num_total += 1
        #         num_correct_norm += int(pred_norm == label)
        #     acc_norm = num_correct_norm / num_total
        #     print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        #     with open(log_file, "a") as f:
        #         f.write(f"{step} hella {acc_norm:.4f}\n")

        # =================================================================
        # SAMPLE GENERATION
        # =================================================================
        # Periodically generate text samples to monitor model quality.
        # Uses top-k sampling for diverse, coherent outputs.
        generate_samples = cfg['evaluation']['generate_samples']
        if generate_samples and ((step > 0 and step % sample_interval == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = cfg['evaluation']['num_samples']
            max_length = cfg['evaluation']['sample_length']
            tokens = enc.encode("Once upon a time,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B, T)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(xgen)  # (B, T, vocab_size)
                    logits = logits[:, -1, :]  # (B, vocab_size) - last token predictions
                    probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
                    # Top-k sampling for diverse generation
                    top_k = cfg['evaluation'].get('top_k', 50)
                    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # (B, top_k)
                    # Sample from top-k distribution
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # Get corresponding token indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # Append to sequence
                    xgen = torch.cat((xgen, xcol), dim=1)  # (B, T+1)
            # Print generated samples
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"sample {i}: {decoded}")

        # =================================================================
        # TRAINING STEP WITH GRADIENT ACCUMULATION
        # =================================================================
        # Performs forward and backward passes with gradient accumulation,
        # gradient clipping, learning rate scheduling, and parameter updates.
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        # Determine dtype for mixed precision training
        dtype_str = cfg['device'].get('dtype', 'bfloat16')
        if device_type == "mps":
            dtype = torch.float16  # MPS doesn't support bfloat16 well
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        # Gradient accumulation loop
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()  # (B, T), (B, T)
            # Async GPU transfer with non_blocking for pinned memory
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Mixed precision forward and backward pass
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(x, y)  # (B, T, vocab_size), scalar
            loss = loss / grad_accum_steps  # Scale loss for accumulation
            loss_accum += loss.detach()
            loss.backward()
        
        # Gradient clipping for training stability
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip_norm'])
        
        # Learning rate schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Update parameters
        optimizer.step()
        
        # Synchronize GPU for accurate timing
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        # Log training metrics
        t1 = time.time()
        dt = t1 - t0  # Step time in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    # =====================================================================
    # TRAINING SUMMARY
    # =====================================================================
    print("\n" + "="*80)
    print("Training complete!")
    if checkpoint_tracker:
        print(f"Best checkpoint: {checkpoint_tracker[0][1]} with val_loss: {checkpoint_tracker[0][0]:.4f}")
    else:
        print("No checkpoints were saved during training.")
    print("="*80 + "\n")

# =============================================================================
# TESTING LOOP WITH ATTENTION VISUALIZATION
# =============================================================================
if test_flag and not train_flag:
    """Testing mode: Load checkpoint, evaluate, and visualize attention patterns.
    
    Performs:
    1. Load model from checkpoint
    2. Compute test set loss and perplexity
    3. Generate sample completions from test prompts
    4. Extract and visualize attention patterns from test examples
    5. Save visualizations and results to experiment directory
    """
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "="*80)
    print("Starting Testing with Attention Visualization...")
    print("="*80 + "\n")
    
    # Create test results directory
    test_results_dir = dirs['exp_dir'] / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    test_log_file = str(test_results_dir / "test_results.txt")
    
    # Load test examples from JSON
    test_examples_file = cfg['data'].get('test_examples_file', 'test_examples.json')
    test_json_path = os.path.join(cfg['data']['data_dir'], test_examples_file)
    print(f"Loading test examples from: {test_json_path}")
    
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(
            f"Test examples file not found: {test_json_path}\n"
            f"Please ensure you have run babylm_data.py to generate the test_examples.json file."
        )
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_examples = json.load(f)
    
    print(f"Loaded {len(test_examples)} test examples\n")
    
    # Validate JSON structure
    if test_examples and len(test_examples) > 0:
        required_keys = {'id', 'text', 'tokens'}
        first_example_keys = set(test_examples[0].keys())
        if not required_keys.issubset(first_example_keys):
            raise ValueError(
                f"Invalid test_examples.json format. Each example must have keys: {required_keys}\n"
                f"Found keys: {first_example_keys}"
            )
        print(f"Validated test examples format: {first_example_keys}")
    
    with open(test_log_file, "w") as f:
        f.write(f"Test Results for checkpoint: {checkpoint_path}\n")
        f.write(f"Step: {checkpoint['step']}\n")
        f.write(f"Number of test examples: {len(test_examples)}\n")
        f.write("="*80 + "\n\n")
    
    # =====================================================================
    # ATTENTION EXTRACTION HELPER
    # =====================================================================
    def extract_attention_patterns(model, tokens_tensor, device):
        """Extract attention weights from all layers and heads.
        
        Manually computes attention patterns by running forward pass through
        each transformer block and extracting Q, K, V projections.
        
        Args:
            model: GPT model to extract attention from.
            tokens_tensor: Input token indices of shape (B, T).
            device: Device to run computation on.
            
        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (B, n_head, T, T).
        """
        model.eval()
        attention_patterns = []
        
        with torch.no_grad():
            tokens_tensor = tokens_tensor.to(device)
            B, T = tokens_tensor.size()
            
            # Forward pass through embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = model.transformer['wpe'](pos)
            tok_emb = model.transformer['wte'](tokens_tensor)
            x = tok_emb + pos_emb
            
            # Go through each layer and extract attention
            for layer_idx, block in enumerate(model.transformer.h):
                # Get attention patterns from this block
                x_norm = block.ln_1(x)
                
                # Compute attention manually to get the patterns
                qkv = block.attn.c_attn(x_norm)
                q, k, v = qkv.split(block.attn.n_embd, dim=2)
                
                # Reshape for multi-head attention
                q = q.view(B, T, block.attn.n_head, block.attn.head_size).transpose(1, 2)
                k = k.view(B, T, block.attn.n_head, block.attn.head_size).transpose(1, 2)
                v = v.view(B, T, block.attn.n_head, block.attn.head_size).transpose(1, 2)
                
                # Compute attention scores
                att_scores = (q @ k.transpose(-2, -1)) / (block.attn.head_size ** 0.5)
                
                # Apply causal mask
                causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                att_scores = att_scores.masked_fill(causal_mask, float('-inf'))
                
                # Softmax to get attention weights
                att_weights = torch.softmax(att_scores, dim=-1)
                
                attention_patterns.append(att_weights.cpu())
                
                # Continue forward pass
                attn_output = (att_weights @ v).transpose(1, 2).contiguous().view(B, T, block.attn.n_embd)
                attn_output = block.attn.c_proj(attn_output)
                x = x + attn_output
                x = x + block.mlp(block.ln_2(x))
        
        return attention_patterns
    
    # =====================================================================
    # ATTENTION VISUALIZATION HELPER
    # =====================================================================
    def plot_layer_attention(attention_weights, tokens_text, layer_idx, save_path):
        """Plot all attention heads for a single layer in one image.
        
        Creates a grid of heatmaps, one per attention head, showing which
        tokens attend to which other tokens.
        
        Args:
            attention_weights: Attention weights of shape (n_head, T, T).
            tokens_text: List of token strings for axis labels.
            layer_idx: Layer index for plot title.
            save_path: Path to save the visualization image.
        """
        n_heads = attention_weights.shape[0]
        T = attention_weights.shape[1]
        
        # Limit sequence length for visualization
        max_len = min(T, 50)
        attention_weights = attention_weights[:, :max_len, :max_len]
        tokens_text = tokens_text[:max_len]
        
        # Calculate grid dimensions
        n_cols = 4
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for head_idx in range(n_heads):
            row = head_idx // n_cols
            col = head_idx % n_cols
            ax = axes[row, col]
            
            head_attention = attention_weights[head_idx].numpy()
            
            # Plot heatmap
            sns.heatmap(head_attention, 
                        cmap='viridis', 
                        xticklabels=tokens_text if max_len <= 20 else False,
                        yticklabels=tokens_text if max_len <= 20 else False,
                        cbar=True,
                        square=True,
                        ax=ax,
                        cbar_kws={'shrink': 0.8})
            
            ax.set_title(f'Head {head_idx}', fontsize=10)
            if max_len <= 20:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
        
        # Hide empty subplots
        for head_idx in range(n_heads, n_rows * n_cols):
            row = head_idx // n_cols
            col = head_idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Layer {layer_idx} - All Attention Heads', fontsize=14, y=1.00)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # =====================================================================
    # ATTENTION VISUALIZATION GENERATION
    # =====================================================================
    # Process test examples and generate attention heatmaps for each layer.
    # Visualizations show how each attention head attends across the sequence.
    num_examples_to_viz = cfg['testing'].get('num_test_samples', 10)
    max_seq_viz = cfg['testing'].get('test_sample_length', 512)
    
    print("Generating attention visualizations for test examples...")
    print(f"Number of examples to visualize: {num_examples_to_viz}")
    print(f"Max sequence length: {max_seq_viz}")
    print(f"Saving to: {dirs['attention_dir']}/\n")
    
    for example_idx, example in enumerate(test_examples[:num_examples_to_viz]):
        example_id = example['id']
        text = example['text']
        tokens = example['tokens']
        
        print(f"Processing example {example_idx + 1}/{num_examples_to_viz} (ID: {example_id})")
        print(f"  Text length: {len(text)} chars")
        print(f"  Token count: {len(tokens)}")
        
        # Create directory for this example's attention plots
        example_attention_dir = dirs['attention_dir'] / str(example_id)
        example_attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tokens to tensor (limit to max_seq_viz)
        tokens_to_viz = tokens[:max_seq_viz]
        tokens_tensor = torch.tensor(tokens_to_viz, dtype=torch.long).unsqueeze(0)
        
        # Get token strings for labels (decode tokens)
        tokens_text = []
        for t in tokens_to_viz:
            try:
                decoded = enc.decode([t]) if t < 50257 else f"<{t}>"
                # Replace newlines and long strings for readability
                decoded = decoded.replace('\n', '\\n')[:10]
                tokens_text.append(decoded)
            except:
                tokens_text.append(f"<{t}>")
        
        # Extract attention patterns
        attention_patterns = extract_attention_patterns(model, tokens_tensor, device)
        
        # Plot each layer
        for layer_idx, layer_attention in enumerate(attention_patterns):
            layer_attention = layer_attention.squeeze(0)  # Remove batch dimension (n_head, T, T)
            
            save_path = example_attention_dir / f"layer{layer_idx:02d}.png"
            plot_layer_attention(layer_attention, tokens_text, layer_idx, str(save_path))
        
        print(f"  Saved {len(attention_patterns)} layer plots to: {example_attention_dir}/")
    
    print(f"\nAttention visualization complete!")
    print("="*80 + "\n")
    
    # =====================================================================
    # TEST SET EVALUATION
    # =====================================================================
    # Compute loss and perplexity on full test set.
    model.eval()
    test_loader.reset()
    
    # Determine dtype for mixed precision
    dtype_str = cfg['device'].get('dtype', 'bfloat16')
    if device_type == "mps":
        dtype = torch.float16  # MPS doesn't support bfloat16 well
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print("Evaluating on test set...")
    with torch.no_grad():
        test_loss_accum = 0.0
        num_test_batches = len(test_loader.tokens) // (B * T)
        print(f"Total test batches: {num_test_batches}")
        
        if num_test_batches > 0:
            for batch_idx in range(num_test_batches):
                x, y = test_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(x, y)
                test_loss_accum += loss.detach()
                
                if (batch_idx + 1) % 100 == 0:
                    avg_loss_so_far = test_loss_accum.item() / (batch_idx + 1)
                    print(f"Batch {batch_idx + 1}/{num_test_batches} | avg loss: {avg_loss_so_far:.4f}")
        
            final_test_loss = test_loss_accum.item() / num_test_batches
            print(f"\n{'='*80}")
            print(f"Final Test Loss: {final_test_loss:.4f}")
            print(f"Test Perplexity: {math.exp(final_test_loss):.4f}")
            print(f"{'='*80}\n")
        else:
            print("Warning: No test batches available (dataset too small).")
            final_test_loss = 0.0
    
    # Log results
    with open(test_log_file, "a") as f:
        f.write(f"Number of test batches: {num_test_batches}\n")
        if num_test_batches > 0:
            f.write(f"Final test loss: {final_test_loss:.4f}\n")
            f.write(f"Test perplexity: {math.exp(final_test_loss):.4f}\n\n")
        else:
            f.write("Warning: No test batches available.\n\n")
    
    # =====================================================================
    # SAMPLE TEXT GENERATION
    # =====================================================================
    # Generate completions from test prompts to qualitatively evaluate model.
    # Uses top-k sampling for diverse, coherent outputs.
    print("Generating sample completions...")
    model.eval()
    num_samples = cfg['testing'].get('num_test_samples', 5)
    sample_prompts = cfg['testing'].get('test_prompts', [
        "Once upon a time,",
        "The little boy",
        "In the garden,",
        "She looked at the",
        "They went to the"
    ])
    
    with open(test_log_file, "a") as f:
        f.write("="*80 + "\n")
        f.write("Sample Generations:\n")
        f.write("="*80 + "\n\n")
    
    for i, prompt in enumerate(sample_prompts[:num_samples]):
        print(f"\nPrompt {i+1}: {prompt}")
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)
        
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + i)
        
        max_gen_length = cfg['testing'].get('test_sample_length', 512)
        top_k = cfg['testing'].get('top_k', 50)
        with torch.no_grad():
            # Autoregressive generation loop
            while tokens.size(1) < max_gen_length:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, _ = model(tokens)  # (1, T, vocab_size)
                logits = logits[:, -1, :]  # (1, vocab_size) - last token predictions
                probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # (1, top_k)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (1, 1)
                xcol = torch.gather(topk_indices, -1, ix)  # (1, 1)
                tokens = torch.cat((tokens, xcol), dim=1)  # (1, T+1)
        
        generated_text = enc.decode(tokens[0].tolist())
        print(f"Generated: {generated_text}")
        
        with open(test_log_file, "a") as f:
            f.write(f"Prompt {i+1}: {prompt}\n")
            f.write(f"Generated: {generated_text}\n\n")
    
    print(f"\n{'='*80}")
    print(f"Testing complete! Results saved to {test_log_file}")
    print(f"{'='*80}\n")
    print("Note: Add attention plotting and other analysis functions here as needed.")
