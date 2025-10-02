# Create a new corrected script
#!/bin/bash
# CPU-optimized testing script for TFLOP

if [ -z "$1" ]; then
    echo "Usage: bash cpu_test_pubtabnet.sh <checkpoint_path>"
    exit 1
fi

CHECKPOINT_PATH=$1
AUX_JSON=$2
AUX_PKL=$4
AUX_IMG_DIR=$3

echo "=== TFLOP CPU Testing Script ==="
echo "Testing checkpoint: $CHECKPOINT_PATH"

# Set CPU environment
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Running inference on 5 samples..."

python test.py \
    --tokenizer_name_or_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_989 \
    --model_name_or_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_989 \
    --exp_config_path config/exp_configs/cpu_general_exp.yaml \
    --model_config_path config/exp_configs/cpu_data_pubtabnet.yaml \
    --checkpoint_path $CHECKPOINT_PATH \
    --aux_json_path         $AUX_JSON \
    --aux_img_path          $AUX_IMG_DIR \
    --aux_rec_pkl_path      $AUX_PKL \
    --batch_size 1 \
    --save_dir inference_results \
    --max_samples 5 \
    --use_validation



# Make it executable

