#!/bin/bash

# Define ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print messages in color
function print_message() {
    echo -e "${1}${2}${NC}"
}

# Save the current directory
CURRENT_DIR=$(pwd)

# Check if .venv directory exists
if [ ! -d ".venv" ]; then
    print_message $GREEN "Virtual environment not found. Creating one..."
    python3 -m venv .venv --system-site-packages || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }
fi

# Activate the virtual environment
source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }

# Install required Python packages, using the path from $CURRENT_DIR
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt" || { print_message $RED "Failed to install Python packages. Exiting."; exit 1; }
else
    print_message $RED "requirements.txt not found in $CURRENT_DIR. Exiting."
    exit 1
fi

# Install extra system/Python deps before model download
#if [ -f "python3-dev_3.10.13-r0_arm64.deb" ]; then
#    dpkg -i python3-dev_3.10.13-r0_arm64.deb || { print_message $RED "Failed to install python3-dev .deb. Exiting."; exit 1; }
#else
#    print_message $YELLOW "python3-dev_3.10.13-r0_arm64.deb not found. Skipping dpkg install."
#fi

# Install tsuki_onnx from Git (correct pip VCS syntax)
pip install "tsuki_onnx @ git+https://github.com/moonshine-ai/tsuki.git" || { print_message $RED "Failed to install tsuki_onnx. Exiting."; exit 1; }


# Download models and pre-generate TTS
python initialize.py || { print_message $YELLOW "Failed to download models.";}

# Print completion message
print_message $GREEN "Setup complete. Run the following commands to start demo:\n"
print_message $BLUE "source .venv/bin/activate\npython assistant.py"
