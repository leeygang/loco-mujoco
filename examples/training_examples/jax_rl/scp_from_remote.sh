#!/bin/bash

# Check if filename argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 myfile.txt"
    exit 1
fi

# Get the filename from the first argument
FILENAME="$1"

# Check if file exists
if [ ! -e "$FILENAME" ]; then
    echo "Error: File '$FILENAME' does not exist"
    exit 1
fi

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"
REMOTE_PATH="~/projects/loco-mujoco/examples/training_examples/jax_rl/"

# Determine if it's a directory or file
if [ -d "$FILENAME" ]; then
    echo "Copying directory '$FILENAME' from $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$FILENAME" .
else
    echo "Copying file '$FILENAME' from $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    scp  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$FILENAME" .
fi

# Check if scp was successful
if [ $? -eq 0 ]; then
    echo "Transfer completed successfully!"
else
    echo "Transfer failed!"
    exit 1
fi
