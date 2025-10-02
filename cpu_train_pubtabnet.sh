#!/bin/bash
# CPU-optimized training script for TFLOP
# Usage: bash cpu_train_pubtabnet.sh

echo "=== TFLOP CPU Training Script ==="
echo "Optimized for Intel i7 + 16GB RAM"
echo

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Limit PyTorch threads and memory allocation
export PYTORCH_CPU_ALLOC_CONF="max_split_size_mb:512"

# Disable xFormers warnings since we're using CPU
export XFORMERS_MORE_DETAILS=0

echo "CPU Environment configured:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  PYTORCH_CPU_ALLOC_CONF: $PYTORCH_CPU_ALLOC_CONF"
echo

# Check available memory
echo "System resources:"
echo "  CPU cores: $(nproc)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo

# Check if required config files exist
if [ ! -f "config/exp_configs/cpu_general_exp.yaml" ]; then
    echo "❌ Error: cpu_general_exp.yaml not found!"
    echo "Please run: bash integrate_cpu_optimizations.sh first"
    exit 1
fi

if [ ! -f "config/exp_configs/cpu_data_pubtabnet.yaml" ]; then
    echo "❌ Error: cpu_data_pubtabnet.yaml not found!"
    echo "Please run: bash integrate_cpu_optimizations.sh first"
    exit 1
fi

# Check if reduced dataset exists
if [ ! -f "TFLOP-dataset/meta_data/dataset_train.jsonl" ]; then
    echo "❌ Error: Reduced dataset not found!"
    echo "Please run: cd TFLOP-dataset/meta_data && python ../../ultra_simple_reduce.py"
    exit 1
fi

# Create results directory
mkdir -p results/cpu_experiments
mkdir -p logs/cpu_experiments

echo "✅ All prerequisites checked successfully"
echo

# Run training with correct TFLOP arguments
echo "Starting TFLOP training with CPU optimizations..."
echo "Using reduced dataset (1%) for feasible CPU training"
echo

python train.py \
    --exp_config config/exp_configs/cpu_general_exp.yaml \
    --data_config config/exp_configs/cpu_data_pubtabnet.yaml \
    --resume_from_checkpoint_path TFLOP/results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_989
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved to: results/cpu_experiments"
    echo "📊 Logs saved to: logs/cpu_experiments"
    echo "📈 View logs: tensorboard --logdir logs/cpu_experiments"
else
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
