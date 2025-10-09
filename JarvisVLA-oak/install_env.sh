#!/bin/bash

# Installation script for JarvisVLA-oak in oak_env environment

echo "Installing JarvisVLA-oak package in editable mode..."

cd /home/minjune/JarvisVLA-oak

# Install Java if not already installed
echo "Checking Java installation..."
if ! command -v java &> /dev/null; then
    echo "Installing OpenJDK 8..."
    conda install --channel=conda-forge openjdk=8 -y
else
    echo "Java already installed"
fi

# Install package in editable mode
echo "Installing package dependencies..."
pip install -e .

echo ""
echo "Installation complete!"
echo ""
echo "Testing installation..."
python -c "import minestudio; print('✓ minestudio imported successfully')"
python -c "from jarvisvla.evaluate.env_helper.craft_agent import CraftWorker; print('✓ CraftWorker imported successfully')"

echo ""
echo "You can now run the evaluation scripts:"
echo "  ./test_oak_log_single.sh"
echo "  ./run_oak_log_100.sh"
