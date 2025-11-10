"""
TinyStories dataset evaluation
https://huggingface.co/datasets/roneneldan/TinyStories
Evaluates the model's perplexity on TinyStories validation set.
"""

import os
import torch
import tiktoken
from datasets import load_dataset
from torch.nn import functional as F

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories_cache")

enc = tiktoken.get_encoding("gpt2")

def download_dataset():
    """Downloads TinyStories dataset"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    print("Loading TinyStories validation dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", cache_dir=DATA_CACHE_DIR)
    return dataset

def iterate_examples(max_examples=None):
    """
    Iterate through TinyStories validation examples.
    Args:
        max_examples: Maximum number of examples to process (None for all)
    """
    dataset = download_dataset()
    count = 0
    for example in dataset:
        if max_examples is not None and count >= max_examples:
            break
        yield example
        count += 1

def evaluate_perplexity(model, device, device_type, max_examples=100, max_length=512):
    """
    Evaluate model perplexity on TinyStories validation set.
    
    Args:
        model: The GPT model to evaluate
        device: Device to run evaluation on
        device_type: 'cuda' or 'cpu'
        max_examples: Number of examples to evaluate
        max_length: Maximum sequence length to process
    
    Returns:
        avg_loss: Average loss across all examples
        perplexity: Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_examples = 0
    
    print(f"Evaluating on TinyStories (max {max_examples} examples)...")
    
    for example in iterate_examples(max_examples=max_examples):
        text = example['text']
        
        # Tokenize the text
        tokens = enc.encode(text)
        if len(tokens) < 2:
            continue
            
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens_tensor[..., :-1], tokens_tensor[..., 1:])
        
        # Accumulate loss
        total_loss += loss.item() * (len(tokens) - 1)
        total_tokens += (len(tokens) - 1)
        num_examples += 1
    
    if total_tokens == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"Evaluated {num_examples} examples, {total_tokens} tokens")
    print(f"Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    return avg_loss, perplexity

if __name__ == "__main__":
    # Test the dataset loading
    print("Testing TinyStories dataset loading...")
    count = 0
    for example in iterate_examples(max_examples=5):
        print(f"\nExample {count + 1}:")
        print(f"Text: {example['text'][:200]}...")
        count += 1
    print(f"\nSuccessfully loaded {count} examples")
