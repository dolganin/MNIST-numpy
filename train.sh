#!/bin/bash

# MNIST training script using pure numpy and pandas with requirements hash check

VENV_DIR="venv"
HASH_FILE=".requirements.hash"
REQ_FILE="requirements.txt"

echo "=============================="
echo "     MNIST PROJECT BOOTSTRAP  "
echo "=============================="

# 1. Check and create virtual environment if necessary
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
else
    echo "Creating virtual environment '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# 3. Check if requirements.txt has changed
if [ -f "$REQ_FILE" ]; then
    NEW_HASH=$(sha256sum "$REQ_FILE" | awk '{print $1}')
    OLD_HASH=$(cat "$HASH_FILE" 2>/dev/null)

    if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "requirements.txt has changed or hash file is missing."
        echo "Installing dependencies..."

        pip install --upgrade pip
        pip install -r "$REQ_FILE"

        echo "$NEW_HASH" > "$HASH_FILE"
        echo "Dependencies installed and hash updated."
    else
        echo "requirements.txt unchanged. Skipping dependency installation."
    fi
else
    echo "requirements.txt not found. Please create this file."
    deactivate
    exit 1
fi

# 4. Run training script
echo "Starting training process..."
python3 main.py

# 5. Deactivate virtual environment
echo "Deactivating virtual environment."
deactivate

echo "Done. Check saved model weights and metrics in the project directory."
