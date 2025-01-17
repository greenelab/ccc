#!/bin/bash

# Run this script from the root of the repository:
# bash ./scripts/run_tests.sh

# Setup environment
source ./scripts/setup_dev.sh

# Install cccgpu with the cuda extension module
echo -e "\033[34mInstalling cccgpu with the cuda extension module...\033[0m"
pip install .

# Run pytest
echo -e "\033[34mRunning Python tests...\033[0m"
pytest -rs --color=yes ./tests/ --ignore ./tests/gpu/excluded

# Run C++ tests
echo -e "\033[34mRunning C++ tests...\033[0m"
for test in ./build/test_*; do
    echo "Running $test..."
    ./$test
done

# Uninstall cccgpu
echo -e "\033[34mUninstalling cccgpu...\033[0m"
pip uninstall cccgpu -y
