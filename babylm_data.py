"""
BabyLM dataset (for pretraining)
https://huggingface.co/datasets/BabyLM-community/babylm-eng
Downloads and tokenizes the data and saves to disk as train, val, and test files.
Run simply as:
$ python tinystories_data.py
Will save files to the local directory "babylm_data".
"""

import os
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
# IMPORTANT: Add your Hugging Face token here (required for gated datasets)
# Get your token from: https://huggingface.co/settings/tokens
# Make sure it's a "Read" token (write access not needed)
HF_TOKEN = "Enter you token"  # Replace with your actual token

# Alternative: Use huggingface-cli login instead of hardcoding token
# Run in terminal: huggingface-cli login
# Then set USE_CLI_LOGIN = True
USE_CLI_LOGIN = False

local_dir = "babylm_data"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
print("Loading BabyLM dataset...")
print("Dataset: BabyLM-community/babylm-eng")
print()

# Check if token is provided
if USE_CLI_LOGIN:
    print("Using huggingface-cli login authentication...")
    use_token = True  # Will use token from HF_HOME
elif HF_TOKEN == "YOUR_HF_TOKEN_HERE" or not HF_TOKEN:
    print("=" * 80)
    print("ERROR: No Hugging Face token provided!")
    print("=" * 80)
    print()
    print("To access this gated dataset, you need to:")
    print("1. Go to https://huggingface.co/datasets/BabyLM-community/babylm-eng")
    print("2. Click 'Access repository' and accept the terms")
    print("3. Get your token from https://huggingface.co/settings/tokens")
    print("   - Create a new token with 'Read' access")
    print("4. Add the token to HF_TOKEN variable in this script")
    print()
    print("Alternative: Run 'huggingface-cli login' in terminal and set USE_CLI_LOGIN=True")
    print("=" * 80)
    exit(1)
else:
    use_token = HF_TOKEN.strip()
    print("Using provided HF token for authentication...")
    print(f"Token starts with: {use_token[:10]}...")
    print()

try:
    # Try loading the dataset - it might have different split configurations
    dataset = load_dataset("BabyLM-community/babylm-eng", token=use_token)
    print(f"Available splits: {list(dataset.keys())}")
    
    # Get the full dataset from available splits
    from datasets import concatenate_datasets
    
    # Concatenate all available splits
    all_splits = []
    for split_name in dataset.keys():
        print(f"Loading split: {split_name} with {len(dataset[split_name])} examples")
        all_splits.append(dataset[split_name])
    
    if len(all_splits) > 1:
        full_ds = concatenate_datasets(all_splits)
    else:
        full_ds = all_splits[0]
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying alternative loading method...")
    # Alternative: load without specifying split
    full_ds = load_dataset("BabyLM-community/babylm-eng", split="train", token=use_token)

print(f"Total examples in dataset: {len(full_ds)}")

# Split the dataset: 89% train, 9% val, 1% test
total_size = len(full_ds)
train_size = int(0.89 * total_size)
val_size = int(0.09 * total_size)
test_size = total_size - train_size - val_size

print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Create the splits using dataset's select method
train_ds = full_ds.select(range(train_size))
val_ds = full_ds.select(range(train_size, train_size + val_size))
test_ds = full_ds.select(range(train_size + val_size, total_size))

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    
    # Try to get text from various possible field names
    if isinstance(doc, dict):
        text = doc.get("text", doc.get("content", doc.get("sentence", "")))
    elif isinstance(doc, str):
        text = doc
    else:
        text = str(doc)
    
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16, text

# tokenize train documents
print("Tokenizing train documents...")
train_tokens = []
for doc in tqdm(train_ds):
    tokens, _ = tokenize(doc)
    train_tokens.extend(tokens)

# tokenize validation documents
print("Tokenizing validation documents...")
val_tokens = []
for doc in tqdm(val_ds):
    tokens, _ = tokenize(doc)
    val_tokens.extend(tokens)

# tokenize test documents
print("Tokenizing test documents...")
test_tokens = []
test_examples = []
for idx, doc in enumerate(tqdm(test_ds)):
    tokens, text_content = tokenize(doc)
    test_tokens.extend(tokens)
    # Save test examples for later use
    test_examples.append({
        'id': idx,
        'text': text_content,
        'tokens': tokens.tolist()
    })

# convert to numpy arrays
print("Converting to numpy arrays...")
train_tokens_np = np.array(train_tokens, dtype=np.uint16)
val_tokens_np = np.array(val_tokens, dtype=np.uint16)
test_tokens_np = np.array(test_tokens, dtype=np.uint16)

print(f"Total train tokens: {len(train_tokens_np):,}")
print(f"Total val tokens: {len(val_tokens_np):,}")
print(f"Total test tokens: {len(test_tokens_np):,}")

# save the numpy files
print("Saving numpy files...")
train_filename = os.path.join(DATA_CACHE_DIR, "babylm_train.npy")
val_filename = os.path.join(DATA_CACHE_DIR, "babylm_val.npy")
test_filename = os.path.join(DATA_CACHE_DIR, "babylm_test.npy")

np.save(train_filename, train_tokens_np)
np.save(val_filename, val_tokens_np)
np.save(test_filename, test_tokens_np)

print(f"Saved {len(train_tokens_np):,} training tokens to {train_filename}")
print(f"Saved {len(val_tokens_np):,} validation tokens to {val_filename}")
print(f"Saved {len(test_tokens_np):,} test tokens to {test_filename}")

# save test examples to JSON file for later reference
print("Saving test examples to JSON...")
test_json_filename = os.path.join(DATA_CACHE_DIR, "test_examples.json")
with open(test_json_filename, 'w', encoding='utf-8') as f:
    json.dump(test_examples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(test_examples)} test examples to {test_json_filename}")
print("Done!")
