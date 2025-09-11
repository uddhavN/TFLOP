# TFLOP
TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism

# Create a new conda environment with Python 3.9
conda create -n tflop python=3.9
conda activate tflop

# Clone the TFLOP repository
git clone https://github.com/UpstageAI/TFLOP

# Install required packages
cd TFLOP
pip install torch==2.0.1 torchmetrics==1.6.0 torchvision==0.15.2
pip install -r requirements.txt
Download required files
install & login huggingface
reference: https://huggingface.co/docs/huggingface_hub/en/guides/cli

pip install -U "huggingface_hub[cli]"
huggingface-cli login
install git-lfs
sudo apt install git-lfs
git lfs install
download dataset from huggingface
git clone https://huggingface.co/datasets/upstage/TFLOP-dataset
Directory Layout

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
unzip image files
cd TFLOP-dataset
cd images
tar -xvzf train.tar.gz
tar -xvzf validation.tar.gz
tar -xvzf test.tar.gz
download pretrained weights
mkdir pretrain_weights
cd pretrain_weights
git clone --branch official https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2
Data preprocessing
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
Training
bash scripts/training/train_pubtabnet.sh
Evaluation
bash scripts/testing/test_pubtabnet.sh <bin_idx> <total_bin_cnt> <experiment_savedir> <epoch_step>
python evaluate_ted.py --model_inference_pathdir <experiment_savedir>/<epoch_step> \
                       --output_savepath <experiment_savedir>/<epoch_step>

# Example
bash scripts/testing/test_pubtabnet.sh 0 1 results/pubtabnet_experiment/expv1 epoch_29_step_231000
