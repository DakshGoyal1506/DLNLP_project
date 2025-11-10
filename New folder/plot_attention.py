"""
Attention visualization utilities for GPT model.
Plots and saves attention patterns for each head in each layer.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
from train_gpt2 import GPT, GPTConfig

def extract_attention_patterns(model, tokens, device):
    """
    Extract attention patterns from all layers and heads.
    
    Args:
        model: GPT model
        tokens: Input tokens tensor (B, T)
        device: Device to run on
    
    Returns:
        List of attention patterns for each layer, shape: [(B, n_head, T, T), ...]
    """
    model.eval()
    attention_patterns = []
    
    # Hook to capture attention patterns
    def attention_hook(module, input, output):
        # output is the attention output, we need to get the attention weights
        # We'll need to modify the forward pass to return attention weights
        pass
    
    with torch.no_grad():
        tokens = tokens.to(device)
        B, T = tokens.size()
        
        # Forward pass through the model
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer['wpe'](pos)
        tok_emb = model.transformer['wte'](tokens)
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

def plot_attention_head(attention_weights, tokens_text, layer_idx, head_idx, save_dir):
    """
    Plot attention pattern for a single head.
    
    Args:
        attention_weights: Attention weights tensor (T, T)
        tokens_text: List of token strings
        layer_idx: Layer index
        head_idx: Head index
        save_dir: Directory to save the plot
    """
    T = attention_weights.shape[0]
    
    # Limit to reasonable sequence length for visualization
    max_len = min(T, 64)
    attention_weights = attention_weights[:max_len, :max_len]
    tokens_text = tokens_text[:max_len]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(attention_weights.numpy(), 
                cmap='viridis', 
                xticklabels=tokens_text,
                yticklabels=tokens_text,
                cbar=True,
                square=True,
                ax=ax)
    
    ax.set_title(f'Layer {layer_idx}, Head {head_idx} - Attention Pattern', fontsize=14, pad=20)
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(save_dir, f'layer_{layer_idx:02d}_head_{head_idx:02d}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def visualize_all_attention_heads(model, input_text, device, save_dir='attention_plots'):
    """
    Visualize attention patterns for all heads in all layers.
    
    Args:
        model: GPT model
        input_text: Input text string
        device: Device to run on
        save_dir: Directory to save attention plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tokenize input
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(input_text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    # Get token strings for labels
    tokens_text = [enc.decode([t]) for t in tokens]
    
    print(f"Visualizing attention for {len(tokens)} tokens across {model.config.n_layer} layers...")
    
    # Extract attention patterns
    attention_patterns = extract_attention_patterns(model, tokens_tensor, device)
    
    # Plot each head in each layer
    total_plots = model.config.n_layer * model.config.n_head
    plot_count = 0
    
    for layer_idx, layer_attention in enumerate(attention_patterns):
        # layer_attention shape: (1, n_head, T, T)
        layer_attention = layer_attention.squeeze(0)  # (n_head, T, T)
        
        for head_idx in range(model.config.n_head):
            head_attention = layer_attention[head_idx]  # (T, T)
            plot_attention_head(head_attention, tokens_text, layer_idx, head_idx, save_dir)
            plot_count += 1
            
            if plot_count % 10 == 0:
                print(f"Progress: {plot_count}/{total_plots} plots completed")
    
    print(f"\nAll {total_plots} attention plots saved to '{save_dir}/'")

def visualize_attention_summary(model, input_text, device, save_dir='attention_plots'):
    """
    Create summary visualizations showing average attention per layer.
    
    Args:
        model: GPT model
        input_text: Input text string
        device: Device to run on
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Tokenize input
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(input_text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens_text = [enc.decode([t]) for t in tokens]
    
    # Extract attention patterns
    attention_patterns = extract_attention_patterns(model, tokens_tensor, device)
    
    # Create summary plot: average attention per layer
    n_layers = len(attention_patterns)
    max_len = min(tokens_tensor.shape[1], 64)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer_idx, layer_attention in enumerate(attention_patterns):
        if layer_idx >= 12:  # Only plot first 12 layers
            break
            
        # Average across all heads
        avg_attention = layer_attention.squeeze(0).mean(dim=0)[:max_len, :max_len]
        
        ax = axes[layer_idx]
        sns.heatmap(avg_attention.numpy(), 
                    cmap='viridis',
                    xticklabels=tokens_text[:max_len],
                    yticklabels=tokens_text[:max_len],
                    cbar=True,
                    square=True,
                    ax=ax)
        
        ax.set_title(f'Layer {layer_idx} (avg)', fontsize=10)
        ax.set_xlabel('Key', fontsize=8)
        ax.set_ylabel('Query', fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    
    plt.tight_layout()
    summary_file = os.path.join(save_dir, 'attention_summary.png')
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary: {summary_file}")

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model.eval()
    
    # Example text
    example_text = "Once upon a time, there was a little girl who loved to read books."
    
    print(f"\nInput text: {example_text}")
    print(f"Generating attention visualizations...")
    
    # Generate all attention plots
    visualize_all_attention_heads(model, example_text, device, save_dir='attention_plots')
    
    # Generate summary plot
    visualize_attention_summary(model, example_text, device, save_dir='attention_plots')
