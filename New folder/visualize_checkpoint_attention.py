"""
Standalone script to visualize attention patterns from a trained model checkpoint.
Run this script after training to generate attention plots without retraining.

Usage:
    python visualize_checkpoint_attention.py --checkpoint log/model_19073.pt --text "Your text here"
"""

import os
import argparse
import torch
from plot_attention import visualize_all_attention_heads, visualize_attention_summary
from train_gpt2 import GPT, GPTConfig

def main():
    parser = argparse.ArgumentParser(description='Visualize attention patterns from a trained model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--text', type=str, default="Once upon a time, there was a little girl who loved to read books.", 
                        help='Text to visualize attention for')
    parser.add_argument('--output_dir', type=str, default='attention_visualizations', 
                        help='Directory to save attention plots')
    args = parser.parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = GPT(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from step {checkpoint['step']}, val_loss: {checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint specified or file not found. Using untrained model...")
        model = GPT(GPTConfig(vocab_size=50304))
    
    model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nInput text: {args.text}")
    print(f"\nGenerating attention visualizations...")
    print(f"Model has {model.config.n_layer} layers with {model.config.n_head} heads each")
    print(f"Total plots to generate: {model.config.n_layer * model.config.n_head}\n")
    
    # Generate visualizations
    visualize_all_attention_heads(model, args.text, device, save_dir=args.output_dir)
    visualize_attention_summary(model, args.text, device, save_dir=args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"All attention visualizations saved to: {args.output_dir}/")
    print(f"{'='*80}")
    
    # Print some statistics
    print("\nFiles generated:")
    files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.png')])
    print(f"  - {len(files)} PNG files")
    print(f"  - Individual head plots: {len([f for f in files if f.startswith('layer_')])}")
    print(f"  - Summary plots: {len([f for f in files if 'summary' in f])}")

if __name__ == "__main__":
    main()
