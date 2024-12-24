#!/bin/bash

# Determine the path to python3
PYTHON_PATH=$(which python3)

if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: python3 not found in the path."
    exit 1
fi

# Copy autocoder.py to ~/bin/autocoder
cp autocoder.py ~/bin/autocoder

# Set the correct shebang line
sed -i "1s|.*|#!$PYTHON_PATH|" ~/bin/autocoder

# Make the autocoder script executable
chmod +x ~/bin/autocoder

echo "Installation complete. You can now run 'autocoder' from the command line."
