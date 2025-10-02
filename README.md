# TFLOP
TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism optimised for CPU only systems

# Create a new conda environment with Python 3.9
conda create -n tflop python=3.9
conda activate tflop


# Install required packages
cd TFLOP
pip install torch==2.0.1 torchmetrics==1.6.0 torchvision==0.15.2
pip install -r requirements.txt
# Download required files
install & login huggingface
reference: https://huggingface.co/docs/huggingface_hub/en/guides/cli

pip install -U "huggingface_hub[cli]"
huggingface-cli login

install git-lfs
sudo apt install git-lfs
git lfs install

# download dataset from huggingface
git clone https://huggingface.co/datasets/upstage/TFLOP-dataset

# Directory Layout

├── images
│   ├── test.tar.gz
│   ├── train.tar.gz
│   └── validation.tar.gz
├── meta_data
│   ├── erroneous_pubtabnet_data.json
│   ├── final_eval_v2.json
│   └── PubTabNet_2.0.0.jsonl
└── pse_results
    ├── test
    │   └── end2end_results.pkl
    ├── train
    │   ├── detection_results_0.pkl
    │   ├── detection_results_1.pkl
    │   ├── detection_results_2.pkl
    │   ├── detection_results_3.pkl
    │   ├── detection_results_4.pkl
    │   ├── detection_results_5.pkl
    │   ├── detection_results_6.pkl
    │   └── detection_results_7.pkl
    └── val
        └── detection_results_0.pkl
# unzip image files
cd TFLOP-dataset
cd images
tar -xvzf train.tar.gz
tar -xvzf validation.tar.gz
tar -xvzf test.tar.gz

# download pretrained weights
mkdir pretrain_weights
cd pretrain_weights
git clone --branch official https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2
# Data preprocessing
preprocess dataset with pse result
bash scripts/preprocess_data/preprocess_pubtabnet.sh
You can get TFLOP-dataset/meta_data/dataset_train.jsonl, TFLOP-dataset/meta_data/validation.jsonl
TFLOP-dataset
├── images
│   ├── test
│   ├── train
│   ├── validation
├── meta_data
│   ├── dataset_train.jsonl
│   ├── dataset_validation.jsonl
│   ├── erroneous_pubtabnet_data.json
│   ├── final_eval_v2.json
│   └── PubTabNet_2.0.0.jsonl
└── pse_results
    ├── test
    ├── train
    └── val
# Training
bash cpu_train_pubtabnet.sh
# Testing
bash cpu_test_pubtabnet.sh
# inference (for a single image)
 python simple_inference.py \
   --model_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_889 \
   --tokenizer_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_889 \
   --exp_config config/exp_configs/cpu_general_exp.yaml \
  --data_config config/exp_configs/cpu_data_pubtabnet.yaml \
   --image_path TFLOP-dataset/images/validation/PMC555548_003_00.png \
  --pse_pkl TFLOP-dataset/pse_results/val/detection_results_0.pkl \
   --output_file single_result.json
references - https://github.com/UpstageAI/TFLOP
