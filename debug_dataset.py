# debug_dataset.py
import sys
sys.path.append('.')
from tflop.datamodule.datasets.tflop import TFLOPDataset

try:
    dataset = TFLOPDataset(
        dataset_path="./TFLOP-dataset",
        metadata_file="meta_data/dataset_train.jsonl",
        images_dir="images/train",
        image_size=[224, 224],
        max_sequence_length=256
    )
    print(f"Dataset length: {len(dataset)}")
    
    # Test first few samples
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            if sample is None:
                print(f"Sample {i}: None")
            else:
                print(f"Sample {i}: OK - Keys: {sample.keys()}")
        except Exception as e:
            print(f"Sample {i}: Error - {e}")
            
except Exception as e:
    print(f"Dataset creation failed: {e}")
