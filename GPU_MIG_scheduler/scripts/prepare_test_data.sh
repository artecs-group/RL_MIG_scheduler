#!/bin/bash

# URL with the test-data to download
URL="https://github.com/artecs-group/FAR_MIG_scheduler/releases/download/1.0/gpu-rodinia.tar.gz"

# Path to the directory where the data will be stored
DEST_DIR="../data/kernels"

# Create the directory if it does not exist
mkdir -p "$DEST_DIR"

# Download the file from the URL using wget
echo "Downloading file from $URL..."
wget "$URL" -O "test_data.tar.gz"

# Extract the file in the specified directory
echo "Extracting file..."
tar -xzvf test_data.tar.gz -C "$DEST_DIR"

# Remove the .tar.gz file
echo "Removing file .tar.gz..."
rm test_data.tar.gz

# Confirmation message
echo "Process completed successfully!"
