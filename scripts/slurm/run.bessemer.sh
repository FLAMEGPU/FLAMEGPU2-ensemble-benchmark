#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
# 1 GPU
#SBATCH --gres=gpu:1
# 10 CPU cores (1/4 of the ndoe)
#SBATCH --cpus-per-task=10
# 1 GPU's worth of host memory
#SBATCH --mem=32G

# Load modules (with matching gcc versions)
module use /usr/local/modulefiles/staging/eb/all/
module load CUDA/11.0.2-GCC-9.3.0

# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the `build` directory.
cd $PROJECT_ROOT
cd build

# Set FLAMEGPU2_INC_DIR poiinting at the included dependency, relative to the build dir where execution is occuring.
# Long term this should not be requied
export FLAMEGPU2_INC_DIR=_deps/flamegpu2-src/include

# Output some GPU information into the Log
nvidia-smi

# Run the executable.
./bin/Release/ensemble-benchmark
