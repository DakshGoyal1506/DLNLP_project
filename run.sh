#!/bin/bash
# Run GPT-2 Training Script with Configuration
# Usage: ./run.sh <config_name>
# Example: ./run.sh baseline

# Check if config name is provided
if [ -z "$1" ]; then
    echo "Error: No configuration file specified"
    echo "Usage: ./run.sh <config_name>"
    echo "Example: ./run.sh baseline"
    exit 1
fi

# Set config name and path
CONFIG_NAME="$1"
CONFIG_PATH="config/${CONFIG_NAME}.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found: $CONFIG_PATH"
    echo "Available configs:"
    ls config/*.yaml 2>/dev/null || echo "No config files found"
    exit 1
fi

echo "================================================================================"
echo "Starting GPT-2 Training"
echo "Configuration: $CONFIG_NAME"
echo "Config File: $CONFIG_PATH"
echo "================================================================================"
echo ""

# Run the training script
python train_gpt2.py --config "$CONFIG_PATH"

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Training failed with error code: $?"
    echo "================================================================================"
    exit $?
fi

echo ""
echo "================================================================================"
echo "Training completed successfully!"
echo "================================================================================"
