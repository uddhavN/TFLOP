#!/usr/bin/env python3
"""
Simple TFLOP CPU Inference Script - Single Image
Includes pointer_args from PSE results for proper TFLOP inference
"""

import traceback
import numpy as np
import argparse
import json
import os
import pickle
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from omegaconf import OmegaConf


from tflop.model.model.TFLOP import TFLOP
from tflop.model.model.TFLOP_Config import TFLOPConfig
from tflop.utils import resolve_missing_config


def load_and_preprocess_image(image_path, input_size=224):
    """Load and preprocess image for TFLOP"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def simple_inference():
    parser = argparse.ArgumentParser(description='Simple TFLOP CPU Inference - Single Image')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--tokenizer_path', required=True, help='Path to tokenizer directory')
    parser.add_argument('--exp_config', required=True, help='Path to experiment config')
    parser.add_argument('--data_config', required=True, help='Path to data config')
    parser.add_argument('--image_path', required=True, help='Path to single test image')
    parser.add_argument('--pse_pkl', required=True, help='PSE results pickle file')
    parser.add_argument('--output_file', default='single_inference_result.json', help='Output file')
    args = parser.parse_args()


    print("=== Simple TFLOP CPU Inference - Single Image ===")
    # Load configs
    exp_config = OmegaConf.load(args.exp_config)
    data_config = OmegaConf.load(args.data_config)
    print("✓ Configs loaded")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("✓ Tokenizer loaded")


    # Initialize model
    model_config_dict = {
        k: v for k, v in exp_config.items()
        if k in TFLOPConfig.get_member_variables()
    }
    model_config_dict = resolve_missing_config(model_config_dict)
    model = TFLOP(
        config=TFLOPConfig(**model_config_dict),
        tokenizer=tokenizer,
        data_ids=["C-tag"],
    )


    # Load weights
    weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    encoder_weights = {k[len("encoder."):]: v
                       for k, v in state_dict.items()
                       if k.startswith("encoder.")}
    decoder_weights = {k[len("decoder."):]: v
                       for k, v in state_dict.items()
                       if k.startswith("decoder.")}
    model.encoder.load_state_dict(encoder_weights)
    model.decoder.load_state_dict(decoder_weights)
    model.eval().float()
    print("✓ Model loaded and ready")


    # Load PSE results
    with open(args.pse_pkl, 'rb') as f:
        pse_results = pickle.load(f)
    print(f"✓ Loaded PSE results: {len(pse_results)} entries")
    print(f"PSE results type: {type(pse_results)}")


    # Get image filename
    fname = os.path.basename(args.image_path)
    print(f"Processing image: {fname}")
    
    # Prepare result
    input_size = getattr(exp_config, 'input_img_size', 224)
    if isinstance(input_size, dict):
        input_size = max(input_size.values())

    try:
        # Load and preprocess image
        img_tensor = load_and_preprocess_image(args.image_path, input_size).unsqueeze(0)

        # Build prompt tokens
        prompt_ids = tokenizer.encode("<s_start>", return_tensors='pt')

        # Find PSE entry in list (assuming list structure)
        pse_entry = None
        fname_key = os.path.splitext(fname)[0]  # Remove .png extension
        
        for entry in pse_results:
            # Assuming each entry has a 'filename' or 'image_name' field
            if isinstance(entry, dict) and entry.get('file_name') == fname:
                pse_entry = entry
                break
            elif isinstance(entry, dict) and entry.get('file_name') == fname_key:
                pse_entry = entry
                break
            elif isinstance(entry, dict) and entry.get('image_name') == fname:
                pse_entry = entry
                break
            elif isinstance(entry, dict) and entry.get('image_name') == fname_key:
                pse_entry = entry
                break
        
        if pse_entry is None:
            print(f"Available PSE entries (first 5):")
            for i, entry in enumerate(pse_results[:5]):
                if isinstance(entry, dict):
                    print(f"  {i}: {entry.get('file_name', entry.get('image_name', 'No filename'))}")
            raise ValueError(f"No PSE entry found for {fname} (also tried {fname_key})")
        
        print(f"✓ Found PSE entry for {fname}")
        
        # Extract bounding boxes (adjust field name as needed)
        # Extract raw polygons
        bboxes_raw = pse_entry.get('bboxes', pse_entry.get('bbox', pse_entry.get('coordinates', [])))
        if len(bboxes_raw) == 0:
            raise ValueError(f"No bboxes found in PSE entry for {fname}")
        # Convert 8-point polygons to 4-point [x_min, y_min, x_max, y_max]
        bboxes = []
        rects = []
        input_size = 224
        
        for poly in bboxes_raw:
            pts = np.array(poly).reshape(-1, 2)
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

            # Clamp to valid range [0, input_size - 1]
            x_min = int(np.clip(x_min, 0, input_size - 1))
            x_max = int(np.clip(x_max, 0, input_size - 1))
            y_min = int(np.clip(y_min, 0, input_size - 1))
            y_max = int(np.clip(y_max, 0, input_size - 1))

            rects.append([x_min, y_min, x_max, y_max])

        bboxes = rects

        # Now convert to tensor of shape (1, num_boxes, 4)
        # Convert to tensor
        if isinstance(bboxes, list):
            bboxes = torch.tensor(bboxes, dtype=torch.long)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.long)
            
        bbox = bboxes.unsqueeze(0)
        pointer_args = {
            "coord_input_idx": bbox,
            "coord_input_length": torch.tensor([bbox.shape[1]])
        }
        if bbox.shape[1] == 0:
            raise ValueError(f"No valid bounding boxes found for image {fname}")

        print(f"✓ Using {bbox.shape[1]} bounding boxes")
        num_boxes = bbox.shape[1]
        model.config.bbox_token_cnt = num_boxes
        # Run inference
        try:
            with torch.no_grad():
                output = model.inference(
                    image_tensors=img_tensor,
                    prompt_tensors=prompt_ids,
                    return_json=False,
                    return_attentions=False,
                    pointer_args=pointer_args
                )
        except Exception as e:
            print("Inference error traceback:")
            traceback.print_exc()
            raise

        # Decode output
        seq = output.get('output_sequences')
        if seq is not None:
            ids = seq[0]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            result = {'prediction': text, 'status': 'success', 'image_path': args.image_path}
            print(f"✓ Prediction: {text}")
        else:
            result = {'status': 'no_output', 'image_path': args.image_path}
            print("✗ No output generated")
            
    except Exception as e:
        result = {'status': 'error', 'error': str(e), 'image_path': args.image_path}
        print(f"✗ Error: {e}")

    # Save result
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {args.output_file}")


if __name__ == "__main__":
    simple_inference()

'''
python simple_inference.py \
            --model_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_689 --tokenizer_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_689 \--exp_config config/exp_configs/cpu_general_exp.yaml \--data_config config/exp_configs/cpu_data_pubtabnet.yaml  \--image_path TFLOP-dataset/images/validation/PMC519026_006_00.png  \--pse_pkl TFLOP-dataset/pse_results/val/detection_results_0.pkl \  --output_file single_result.json
'''
# python simple_inference.py \
#   --model_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_889 \
#   --tokenizer_path ./results/cpu_experiments/tflop_cpu_experiment/cpu_v1.0/epoch_1_step_889 \
#   --exp_config config/exp_configs/cpu_general_exp.yaml \
#   --data_config config/exp_configs/cpu_data_pubtabnet.yaml \
#   --image_path TFLOP-dataset/images/validation/PMC555548_003_00.png \
#   --pse_pkl TFLOP-dataset/pse_results/val/detection_results_0.pkl \
#   --output_file single_result.json
