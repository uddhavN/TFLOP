import json
import os

import pickle
import os
import torch.nn.functional as F

from PIL import Image
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from tflop.datamodule.preprocess.common_utils import (
    convert_gold_coords,
    generate_filled_html,
    get_cell_pointer_label,
    get_dr_pointer_label,
    int_convert_and_pad_coords,
    rescale_bbox,
    serialize_bbox_top_left_bottom_right,
)
from tflop.datamodule.preprocess.image_utils import prepare_image_tensor


class TFLOPDataset(Dataset):
    def __init__(
        self: "TFLOPDataset",
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        config: DictConfig = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.shuffle_bbox_shuffle_rate = 0.5

        # Run sanity check on implementation status to avoid unexpected runs / errors
        self.implementation_check()

        # Set up necessary variables
        self.set_up_variables()

        # Load dataset
        self.manual_load_dataset()

        # Set up necessary token ids
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

        # Initialize OTSL config
        self.initialize_OTSL_config()

    def implementation_check(self: "TFLOPDataset"):
        """
        Sanity check of implementation status to avoid unexpected runs / behaviour
        """
        if self.split not in ["train", "validation"]:
            raise ValueError(
                "split must be either 'train' or 'validation'. But got %s" % self.split
            )

        # Shuffling of bbox coordinates should only be done in training set
        if self.config.get("shuffle_cell_bbox_rate", 0.0) != 0.0:
            assert (
                self.split == "training"
            ), "shuffle_cell_bbox_rate must be 0.0 when split is not 'training'."

        # OTSL config check
        if not self.config.get("use_OTSL", False):
            raise NotImplementedError("Non OTSL modes are deprecated.")
    def extract_gold_coords(self, html_data):
        gold_coords = []
        if html_data is None:
            return gold_coords

        cells = html_data.get("cells", [])
        for cell in cells:
            bbox = cell.get("bbox")
            if bbox is not None:
                gold_coords.append(bbox)
        return gold_coords

    def set_up_variables(self: "TFLOPDataset"):
        # General
        self.image_path = self.config.get("image_path", None)
        self.meta_data_path = self.config.get("meta_data_path", None)
        dataset_root = self.config.dataset_path
        if self.split == "train":
            meta_file = self.config.train_metadata_file
        elif self.split == "validation":
            meta_file = self.config.val_metadata_file
        else:
            raise ValueError(f"Unknown split: {self.split}")
        full_meta_path = os.path.join(dataset_root, meta_file)
       # Directory containing the JSONL file+        
        self.meta_data_path = os.path.dirname(full_meta_path)
        
        self.prompt_end_token = self.config.get("prompt_end_token", "<s_answer>")
        
        self.input_img_size = (
            self.config.input_size.width,
            self.config.input_size.height,
        )

        # Pointer-specific
        self.bbox_token_cnt = self.config.get("bbox_token_cnt", None)
        self.max_length = self.config.get("max_length", None)
        self.use_cell_bbox = self.config.get("use_cell_bbox", False)
        self.shuffle_cell_bbox_rate = self.config.get("shuffle_cell_bbox_rate", 0.0)

        # Experimental
        self.use_OTSL = True
        self.span_coeff_mode = self.config.get("span_coeff_mode", "proportional")

        self.coeff_tensor_args = {
            "tokenizer": None,
            "rep_mode": None,
            "rowspan_coeff_mode": self.span_coeff_mode,
            "colspan_coeff_mode": self.span_coeff_mode,
        }

        self.coeff_tensor_args["rep_mode"] = "OTSL"

    def manual_load_dataset(self):
    # Use config's metadata file directly
        if self.split == "train":
            meta_data_path = os.path.join(self.config.dataset_path, self.config.train_metadata_file)
        elif self.split == "validation":
            meta_data_path = os.path.join(self.config.dataset_path, self.config.val_metadata_file)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    # Read all lines in metadata file
        all_metadata = open(meta_data_path, "r").readlines()

    # Filter metadata to keep only entries with existing image files
        filtered_metadata = []
        for line in all_metadata:
            data_obj = json.loads(line)
            image_path = os.path.join(self.image_path, self.split, data_obj["file_name"])
            if os.path.isfile(image_path):
                filtered_metadata.append(line)
            else:
                print(f"Warning: Missing image file, skipping {image_path}")

        self.metadata_loaded = filtered_metadata
    def initialize_OTSL_config(self: "TFLOPDataset"):
        """Initialize tokens which signify data cell in OTSL representations."""

        if self.use_OTSL:
            self.data_ids = [
                self.tokenizer.convert_tokens_to_ids(token) for token in ["C-tag"]
            ]
        else:
            raise NotImplementedError("Non OTSL modes are deprecated.")

    def __len__(self: "TFLOPDataset") -> int:
        return len(self.metadata_loaded)

    def __getitem__(self: "TFLOPDataset", idx: int):
        org_img_size = None
        padding_dims = None
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Initialize placeholders
                html = ""
                cell_texts = []

                # Load metadata and sample dict
                data_selected = self.load_sampled_metadata(idx)
                sample = self.build_sample_dictionary(data_selected)
                if sample is None:
                    raise ValueError(f"Sample at idx={idx} is None.")

                # Convert gold coords
                gold_coords = convert_gold_coords(sample["gold_coord"])
                if gold_coords is None:
                    raise ValueError("gold_coords could not be obtained")

                # Prepare input tokens
                input_parser_seq, input_parse, input_ids = self.get_input_parser_seq_and_tok(sample)
                max_input_length = self.max_length - self.bbox_token_cnt
                max_bbox_length = self.bbox_token_cnt

                # Generate pointer labels & rescaled_coords via appropriate function
                if self.use_cell_bbox:
                    pointer_label, pointer_mask_label, rescaled_coords, cell_texts, bbox_tensor = \
                        get_cell_pointer_label(
                            gold_cell_coords=sample["cell_coord"],
                            gold_coord_isFilled=gold_coords["isFilled"],
                            cell_shuffle_rate=self.shuffle_bbox_shuffle_rate,
                            input_ids=input_ids,
                            data_ids=self.data_ids,
                            bbox_token_cnt=self.bbox_token_cnt,
                            coeff_tensor_args=self.coeff_tensor_args,
                        )
                else:
                    pointer_label, pointer_mask_label, rescaled_coords, cell_texts, bbox_tensor = \
                        get_dr_pointer_label(
                            dr_coords=sample["dr_coord"],
                            gold_coord_isFilled=gold_coords["isFilled"],
                            cell_shuffle_rate=self.shuffle_bbox_shuffle_rate,
                            input_ids=input_ids,
                            data_ids=self.data_ids,
                            bbox_token_cnt=self.bbox_token_cnt,
                            org_img_size=org_img_size,
                            new_img_size=self.input_img_size,
                            padding_dims=padding_dims,
                            coeff_tensor_args=self.coeff_tensor_args,
                        )

                # Defensive check
                if pointer_label is None or pointer_mask_label is None:
                    raise RuntimeError(f"Pointer labels not generated for sample idx: {idx}")

                # Pad & convert coordinates
                coords_padding = [
                    self.input_img_size[0] + 3,
                    self.input_img_size[1] + 3,
                    self.input_img_size[0] + 3,
                    self.input_img_size[1] + 3,
                ]
                coords_int_padded = int_convert_and_pad_coords(
                    coords=rescaled_coords,
                    padding_coord=coords_padding,
                    max_length=self.bbox_token_cnt,
                )

                # Convert types
                valid_coord_length = torch.tensor(len(rescaled_coords), dtype=torch.long)
                pointer_label = pointer_label.to(torch.bool)
                pointer_mask_label = pointer_mask_label.to(torch.bool)

                # Pad input_ids
                if input_ids.size(0) < max_input_length:
                    pad_len = max_input_length - input_ids.size(0)
                    pad_tensor = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, pad_tensor], dim=0)
                else:
                    input_ids = input_ids[:max_input_length]

                # Pad pointer_label
                ptr_len_0, ptr_len_1 = pointer_label.shape
                pad_0 = max_input_length - ptr_len_0
                pad_1 = max_bbox_length - ptr_len_1
                pointer_label = F.pad(pointer_label, (0, pad_1, 0, pad_0), value=0)

                # Pad pointer_mask_label
                if pointer_mask_label.size(0) < max_input_length:
                    pad_len = max_input_length - pointer_mask_label.size(0)
                    mask_pad = torch.zeros(pad_len, dtype=pointer_mask_label.dtype)
                    pointer_mask_label = torch.cat([pointer_mask_label, mask_pad], dim=0)
                else:
                    pointer_mask_label = pointer_mask_label[:max_input_length]

                # Pad coords_int_padded
                if coords_int_padded.size(0) < max_bbox_length:
                    pad_len = max_bbox_length - coords_int_padded.size(0)
                    pad_coords = torch.zeros((pad_len, coords_int_padded.size(1)), dtype=coords_int_padded.dtype)
                    coords_int_padded = torch.cat([coords_int_padded, pad_coords], dim=0)
                else:
                    coords_int_padded = coords_int_padded[:max_bbox_length]

                # Load image & html
                img_tensor, org_img_size, padding_dims = self.load_sampled_image(sample)
                html = generate_filled_html(
                    gold_coords["text"], gold_coords["isFilled"], sample["org_html"]
                )

                # Compose chosen bbox tensor
                if self.config.use_RowWise_contLearning or self.config.use_ColWise_contLearning:
                    chosen_bbox_list = []
                    if self.config.use_RowWise_contLearning:
                        chosen_bbox_list.append(bbox_tensor[0])
                    if self.config.use_ColWise_contLearning:
                        chosen_bbox_list.append(bbox_tensor[1])
                    chosen_bbox_tensor = torch.stack(chosen_bbox_list, dim=0)
                else:
                    chosen_bbox_tensor = torch.zeros_like(bbox_tensor[0]).unsqueeze(0)

                # Return train or eval tuple
                if self.split == "train":
                    token_labels = input_ids.clone()
                    token_labels[token_labels == self.tokenizer.pad_token_id] = -100
                    token_labels[: torch.nonzero(token_labels == self.prompt_token_id).sum() + 1] = -100

                    return (
                        img_tensor,
                        input_ids,
                        coords_int_padded,
                        valid_coord_length,
                        token_labels,
                        pointer_label,
                        pointer_mask_label,
                        chosen_bbox_tensor,
                    )
                else:
                    prompt_idx = torch.nonzero(input_ids == self.prompt_token_id).sum()
                    cell_text_concat = "<special_cell_sep>".join(cell_texts)

                    return (
                        img_tensor,
                        input_ids,
                        coords_int_padded,
                        valid_coord_length,
                        prompt_idx,
                        input_parse,
                        pointer_label,
                        pointer_mask_label,
                        html,
                        cell_text_concat,
                        sample["file_name"],
                        chosen_bbox_tensor,
                    )

            except Exception as e:
                print(f"Error processing sample idx={idx}: {e}")
                idx = (idx + 1) % len(self)
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed to load sample after {max_retries} attempts, returning fallback")
                    break

        return self._create_fallback_sample()


    def _create_fallback_sample(self):
        """Create a minimal valid sample when all else fails."""
        img_tensor = torch.zeros((3, self.input_img_size[0], self.input_img_size[1]))
        input_ids = torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - self.bbox_token_cnt))
        coords_int_padded = torch.zeros((self.bbox_token_cnt, 4), dtype=torch.long)
        valid_coord_length = torch.tensor(0)
        chosen_bbox_coeff_tensor = torch.zeros((1, self.bbox_token_cnt, self.bbox_token_cnt))

        if self.split == "train":
            token_pred_labels = torch.full_like(input_ids, -100)
            pointer_label = torch.zeros(self.bbox_token_cnt, dtype=torch.bool)
            pointer_mask_label = torch.zeros(self.bbox_token_cnt, dtype=torch.bool)
            return (
                img_tensor,
                input_ids,
                coords_int_padded,
                valid_coord_length,
                token_pred_labels,
                pointer_label,
                pointer_mask_label,
                chosen_bbox_coeff_tensor,
            )
        else:
            prompt_end_index = torch.tensor(0)
            input_parse = ""
            pointer_label = torch.zeros(self.bbox_token_cnt, dtype=torch.bool)
            pointer_mask_label = torch.zeros(self.bbox_token_cnt, dtype=torch.bool)
            html_with_content = ""
            cell_text_collated = ""
            file_name = "fallback_sample"
            return (
                img_tensor,
                input_ids,
                coords_int_padded,
                valid_coord_length,
                prompt_end_index,
                input_parse,
                pointer_label,
                pointer_mask_label,
                html_with_content,
                cell_text_collated,
                file_name,
                chosen_bbox_coeff_tensor,
            )



    def load_sampled_metadata(self, idx):
        """Given data idx, load corresponding metadata info."""

        sampled_metadata = json.loads(self.metadata_loaded[idx])
        return sampled_metadata

    def generate_OTSL_sequence(self, sample):

        tokens = []
        html = sample.get("html", {})
        cells = html.get("cells", [])
        for cell in cells:
        # Each cell might have a 'tokens' list
            tokens.extend(cell.get("tokens", []))
        return tokens

    def build_sample_dictionary(self, data_selected):
        sample = {}
        image_path = os.path.join(
            self.image_path,
            self.split,
            data_selected["file_name"]
        )

        img = Image.open(image_path)
        sample["image"] = img

        for data_k, data_v in data_selected.items():
            sample[data_k] = data_v
        sample["gold_coord"] = self.extract_gold_coords(sample["html"])
        if self.use_OTSL:
            sample["otsl_seq"] = self.generate_OTSL_sequence(sample)  # implement this method

        return sample

    def get_input_parser_seq_and_tok(self, sample):
        """Get input text sequence and tokens for model."""
        # 1. OTSL sequence
        OTSL_data = "".join(sample["otsl_seq"]) if self.use_OTSL else None

        # 2. tokenize
        input_parse = (
            self.tokenizer.bos_token
            + self.prompt_end_token
            + OTSL_data
            + "</s_answer>"
            + self.tokenizer.eos_token
        )
        input_ids = self.tokenizer(
            input_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return OTSL_data, input_parse, input_ids

    def load_and_prepare_img_tensor(self, sample):
        """Load and prepare image as tensor for model."""
        img_tensor, org_img_size, padding_dims = prepare_image_tensor(
            input_image=sample["image"],
            target_img_size=self.input_img_size,  # (width, height)
            random_padding=self.split == "train",
        )

        return img_tensor, org_img_size, padding_dims

    def rescale_gold_coordinates(self, gold_coords, org_img_size, padding_dims):
        """Rescale gold coordinates."""
        rescale_fn, coord_input = rescale_bbox, gold_coords["coords"]
        rescaled_coords = rescale_fn(
            list_of_coords=coord_input,
            org_img_size=org_img_size,
            new_img_size=self.input_img_size,
            padding_dims=padding_dims,
        )

        return rescaled_coords


class TFLOPTestDataset(Dataset):
    """Simplified test dataset class for TFLOP."""

    def __init__(
        self: "TFLOPTestDataset",
        tokenizer: PreTrainedTokenizer,
        split: str = "test",
        config: DictConfig = None,
        aux_json_path: str = None,
        aux_img_path: str = None,
        aux_rec_pkl_path: str = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.aux_json_path = aux_json_path
        self.aux_img_path = aux_img_path
        self.aux_rec_pkl_path = aux_rec_pkl_path

        # Run sanity check on implementation status to avoid unexpected runs / errors
        self.implementation_check()

        # Set up necessary variables
        self.set_up_variables()

        # Load dataset
        self.manual_load_dataset()

        # Set up necessary token ids
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

        # Initialize OTSL config
        self.initialize_OTSL_config()

    def implementation_check(self: "TFLOPTestDataset"):
        """
        Sanity check of implementation status to avoid unexpected runs / behaviour
        """
        if self.split not in ["validation", "test"]:
            raise ValueError(
                "split must be either validation or test. But got %s" % self.split
            )

        # Shuffling of bbox coordinates should only be done in training set
        assert (
            self.config.get("shuffle_cell_bbox_rate", 0.0) == 0.0
        ), "shuffle_cell_bbox_rate must be 0.0 when not training."

    def set_up_variables(self: "TFLOPTestDataset"):
        # General
        self.prompt_end_token = self.config.get("prompt_end_token", "<s_answer>")
        self.input_img_size = (
            self.config.input_size.width,
            self.config.input_size.height,
        )

        # Pointer-specific
        self.bbox_token_cnt = self.config.get("bbox_token_cnt", None)
        self.max_length = self.config.get("max_length", None)
        self.use_cell_bbox = self.config.get("use_cell_bbox", False)
        self.shuffle_cell_bbox_rate = self.config.get("shuffle_cell_bbox_rate", 0.0)

        # Experimental
        self.use_OTSL = self.config.get("use_OTSL", False)

    def manual_load_dataset(self: "TFLOPTestDataset"):
        """
        Load dataset for evaluation.
        NOTE:
            - If metadata.jsonl path is present, load from there.
            - Else, load from aux_json_path.
        """
        for aux_filepath in [self.aux_json_path, self.aux_rec_pkl_path]:
            assert (
                aux_filepath is not None
            ), "aux_json_path and aux_rec_pkl_path must be provided."
            assert os.path.exists(aux_filepath), (
                "aux_filepath %s does not exist." % aux_filepath
            )
        assert os.path.exists(self.aux_img_path), (
            "aux_img_path %s does not exist." % self.aux_img_path
        )
        self.using_metadata_jsonl = False

        with open(self.aux_json_path, "r") as f:
            self.aux_json = [json.loads(line) for line in f]
        self.aux_rec = pickle.load(open(self.aux_rec_pkl_path, "rb"))

    def initialize_OTSL_config(self: "TFLOPTestDataset"):
        """Initialize tokens which signify data cell in OTSL representations."""

        if self.use_OTSL:
            self.data_ids = [
                self.tokenizer.convert_tokens_to_ids(token) for token in ["C-tag"]
            ]
        else:
            raise NotImplementedError("Non OTSL modes are deprecated.")

    def __len__(self: "TFLOPTestDataset") -> int:

        if self.using_metadata_jsonl:
            return len(self.metadata_loaded)
        else:
            return len(self.aux_json)

    def __getitem__(self: "TFLOPTestDataset", idx: int):
        max_retries = 3  # Fewer retries for test dataset
        retry_count = 0
        gold_coords = None 
        org_img_size = None
        while retry_count < max_retries:
            try:
                # Loading sample data
                sample = self.load_aux_data(idx)
                
                # **VALIDATION**: Check if sample is valid
                if sample is None:
                    raise ValueError(f"load_aux_data returned None for index {idx}")
                
                if "image" not in sample:
                    raise KeyError(f"Missing 'image' key in sample {sample.get('file_name', 'unknown')}")

                # Prepare image tensor
                img_tensor, org_img_size, padding_dims = prepare_image_tensor(
                    input_image=sample["image"],
                    target_img_size=self.input_img_size,  # (width, height)
                    random_padding=False,
                )
                
                # **VALIDATION**: Check image tensor
                if img_tensor is None or img_tensor.numel() == 0:
                    raise ValueError(f"Invalid image tensor for sample {sample.get('file_name', 'unknown')}")

                # Get full html with content & cell-wise coord & cell-wise text
                result = self.get_coord_and_html_with_content(sample, org_img_size, padding_dims)
                
                # **VALIDATION**: Check if get_coord_and_html_with_content returns valid results
                if result is None or len(result) != 3:
                    raise ValueError(f"Invalid result from get_coord_and_html_with_content for sample {sample.get('file_name', 'unknown')}")
                
                html_with_content, rescaled_coords, cell_texts = result
                
                # **VALIDATION**: Check essential data structures
                if rescaled_coords is None:
                    print(f"Warning: rescaled_coords is None for sample {sample.get('file_name', 'unknown')}, using empty list")
                    rescaled_coords = []
                
                if cell_texts is None:
                    print(f"Warning: cell_texts is None for sample {sample.get('file_name', 'unknown')}, using empty list")
                    cell_texts = []
                
                if html_with_content is None:
                    print(f"Warning: html_with_content is None for sample {sample.get('file_name', 'unknown')}, using empty string")
                    html_with_content = ""

                # Coordinate processing
                padding_coord = [
                    self.input_img_size[0] + 3,
                    self.input_img_size[1] + 3,
                    self.input_img_size[0] + 3,
                    self.input_img_size[1] + 3,
                ]
                
                coords_int_padded = int_convert_and_pad_coords(
                    coords=rescaled_coords,
                    padding_coord=padding_coord,
                    max_length=self.bbox_token_cnt,
                )
                valid_coord_length = torch.tensor(len(rescaled_coords))  # (1)

                # Prepare input token ids
                input_parse = self.tokenizer.bos_token + self.prompt_end_token
                input_ids = self.tokenizer(
                    input_parse,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"].squeeze(0)
                input_ids = input_ids[: self.max_length - self.bbox_token_cnt]

                # **VALIDATION**: Check tokenization results
                if input_ids is None or input_ids.numel() == 0:
                    raise ValueError(f"Invalid input_ids for sample {sample.get('file_name', 'unknown')}")

                # Return values
                prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()
                cell_text_collated = "<special_cell_text_sep>".join(cell_texts)

                return (
                    img_tensor,
                    input_ids,
                    coords_int_padded,
                    valid_coord_length,
                    prompt_end_index,
                    html_with_content,
                    cell_text_collated,
                    sample["file_name"],
                )

            except (KeyError, ValueError, IndexError, RuntimeError) as e:
                print(f"Error processing test sample {idx}: {e}")
                print(f"Sample info: {sample.get('file_name', 'unknown') if 'sample' in locals() else 'N/A'}")
                
                # Move to next sample
                idx = (idx + 1) % len(self.data)
                retry_count += 1
                
                if retry_count >= max_retries:
                    print(f"Failed to load valid test sample after {max_retries} attempts")
                    break
                    
                print(f"Retrying with test sample {idx} (attempt {retry_count + 1}/{max_retries})")
                continue
            
            except Exception as e:
                print(f"Unexpected error processing test sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                idx = (idx + 1) % len(self.data)
                retry_count += 1
                
                if retry_count >= max_retries:
                    break
                continue

        # If all retries failed, create a fallback sample for test dataset
        print(f"All retries failed for test sample, creating fallback sample")
        return self._create_test_fallback_sample()

def _create_test_fallback_sample(self):
    """Create a minimal valid test sample when all else fails"""
    # Create minimal valid tensors for test dataset
    img_tensor = torch.zeros((3, self.input_img_size[0], self.input_img_size[1]))
    input_ids = torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - self.bbox_token_cnt))
    coords_int_padded = torch.zeros((self.bbox_token_cnt, 4), dtype=torch.long)
    valid_coord_length = torch.tensor(0)
    prompt_end_index = torch.tensor(0)
    html_with_content = ""
    cell_text_collated = ""
    file_name = "fallback_test_sample"
    
    return (
        img_tensor,
        input_ids,
        coords_int_padded,
        valid_coord_length,
        prompt_end_index,
        html_with_content,
        cell_text_collated,
        file_name,
    )


    def load_aux_data(self: "TFLOPTestDataset", idx: int):
        """
        When not using metadata jsonl, load metadata from aux json, aux det pkl and aux rec pkl.

        Returns:
            sample: dict
                - image: PIL Image
                - full_html_seq: str, full html sequence as string
                - html_type: str, whether the html seq is simple or complex
                - bbox_coords: list of bbox lists # was previously cell_bboxes
                - bbox_texts: list of bbox texts # was previously cell_texts
                - file_name: str, image filename
        """
        assert (
            not self.using_metadata_jsonl
        ), "load_aux_data should only be called when not using metadata jsonl."

        # Get image filenames
        img_filenames = list(self.aux_json.keys())
        img_filenames.sort()
        selected_filename = img_filenames[idx]

        # 1. Load image
        sample = {
            "image": Image.open(os.path.join(self.aux_img_path, selected_filename))
        }

        # 2. Load other data
        sample["full_html_seq"] = self.aux_json[selected_filename]["html"]
        sample["html_type"] = self.aux_json[selected_filename]["type"]

        rec_result = self.aux_rec[selected_filename]
        sample["bbox_coords"] = [list(c["bbox"]) for c in rec_result]
        sample["bbox_texts"] = [c["text"] for c in rec_result]
        sample["file_name"] = selected_filename



        return sample

    def get_coord_and_html_with_content(
        self: "TFLOPTestDataset", sample, org_img_size, padding_dims
    ):
        if self.using_metadata_jsonl:
            gold_coords = convert_gold_coords(sample["gold_coord"])

            html_with_content = generate_filled_html(
                gold_text_list=gold_coords["text"],
                is_cell_filled=gold_coords["isFilled"],
                org_html_list=sample["org_html"],
            )
            if self.use_cell_bbox:
                # rescale coords
                rescaled_coords = self.rescale_gold_coordinates(
                    gold_coords, org_img_size, padding_dims
                )

                # Only retrieved filled cells
                rescaled_coords = [
                    x
                    for i, x in enumerate(rescaled_coords)
                    if gold_coords["isFilled"][i]
                ]
                cell_texts = [
                    x
                    for i, x in enumerate(gold_coords["text"])
                    if gold_coords["isFilled"][i]
                ]

                # serialize the bbox coords and texts
                rescaled_coords, cell_texts, _ = serialize_bbox_top_left_bottom_right(
                    rescaled_coords, cell_texts
                )
                if len(rescaled_coords) > (self.bbox_token_cnt - 1):
                    rescaled_coords = rescaled_coords[: (self.bbox_token_cnt - 1)]
                    cell_texts = cell_texts[: (self.bbox_token_cnt - 1)]

            else:
                filtered_dr_coords = []
                filtered_texts = []

                dr_coord_keys = [int(x) for x in list(sample["dr_coord"].keys())]
                for tmp_idx in sorted(dr_coord_keys):
                    rescaled_coords = rescale_bbox(
                        list_of_coords=sample["dr_coord"][str(tmp_idx)][0],
                        org_img_size=org_img_size,
                        new_img_size=self.input_img_size,
                        padding_dims=padding_dims,
                    )
                    filtered_dr_coords.extend(rescaled_coords)
                    filtered_texts.append(sample["dr_coord"][str(tmp_idx)][2])
                    filtered_texts += [""] * (len(rescaled_coords) - 1)

                filtered_dr_coords, filtered_texts, _ = (
                    serialize_bbox_top_left_bottom_right(
                        filtered_dr_coords, filtered_texts
                    )
                )

                if len(filtered_dr_coords) > (self.bbox_token_cnt - 1):
                    filtered_dr_coords = filtered_dr_coords[: (self.bbox_token_cnt - 1)]
                    filtered_texts = filtered_texts[: (self.bbox_token_cnt - 1)]

                rescaled_coords = filtered_dr_coords
                cell_texts = filtered_texts
        else:
            html_with_content = sample["full_html_seq"]
            rescaled_coords = rescale_bbox(
                list_of_coords=sample["bbox_coords"],
                org_img_size=org_img_size,
                new_img_size=self.input_img_size,
                padding_dims=padding_dims,
            )

            rescaled_coords, cell_texts, _ = serialize_bbox_top_left_bottom_right(
                rescaled_coords, sample["bbox_texts"]
            )
            if len(rescaled_coords) > (self.bbox_token_cnt - 1):
                rescaled_coords = rescaled_coords[: (self.bbox_token_cnt - 1)]
                cell_texts = cell_texts[: (self.bbox_token_cnt - 1)]

        return html_with_content, rescaled_coords, cell_texts

    def rescale_gold_coordinates(self, gold_coords, org_img_size, padding_dims):
        """Rescale gold coordinates."""

        rescale_fn, coord_input = rescale_bbox, gold_coords["coords"]
        rescaled_coords = rescale_fn(
            list_of_coords=coord_input,
            org_img_size=org_img_size,
            new_img_size=self.input_img_size,
            padding_dims=padding_dims,
        )

        return rescaled_coords
