import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
import numpy as np
import h5py
import hdf5plugin # Important: Ensure this is installed for zstd
import os
from tqdm.auto import tqdm
import logging
from torch.utils.data import DataLoader, Dataset as TorchDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Helper dataset class for batch loading
class ImageDataset(TorchDataset):
    def __init__(self, hf_dataset, indices):
        self.dataset = hf_dataset.select(indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return image and all labels needed, convert PIL image later
        item = self.dataset[idx]
        # Ensure image field name is correct ('image' for fairface)
        return {'image': item['image'], 'age': item['age'], 'gender': item['gender'], 'race': item['race'], 'original_index': idx}


def collate_fn(batch, processor):
    """Collate function to handle PIL images and process them."""
    images = [item['image'] for item in batch]
    try:
        # Process images using the Florence-2 processor
        processed_inputs = processor(text=None, images=images, return_tensors="pt")
    except Exception as e:
        log.warning(f"Processor error on batch: {e}. Skipping batch item if needed.")
        # Handle potential errors (e.g., corrupted images) - this example skips
        return None # Or filter the batch

    # Keep other data associated with the batch
    labels = {
        'age': torch.tensor([item['age'] for item in batch], dtype=torch.uint8),
        'gender': torch.tensor([item['gender'] for item in batch], dtype=torch.uint8),
        'race': torch.tensor([item['race'] for item in batch], dtype=torch.uint8),
        'original_index': torch.tensor([item['original_index'] for item in batch])
    }
    return {'processed_inputs': processed_inputs, 'labels': labels}


def extract_and_save_patches(cfg: DictConfig, split_name: str, indices_file: str, hdf5_file: h5py.File):
    """Extracts patches for a given split and saves to HDF5."""
    log.info(f"--- Processing split: {split_name} ---")

    # Load subset indices
    try:
        indices = np.load(indices_file).tolist()
        log.info(f"Loaded {len(indices)} indices for {split_name} from {indices_file}")
    except FileNotFoundError:
        log.error(f"Indices file not found: {indices_file}. Run prepare_subset task first.")
        raise
    except Exception as e:
        log.error(f"Error loading indices from {indices_file}: {e}")
        raise

    if not indices:
        log.warning(f"No indices found for split {split_name}. Skipping.")
        return

    # Load corresponding split from FairFace
    try:
        full_ds = load_dataset("huggingface/fairface", split=split_name)
    except Exception as e:
        log.error(f"Failed to load FairFace {split_name} split: {e}")
        raise

    # Create PyTorch Dataset and DataLoader
    pytorch_ds = ImageDataset(full_ds, indices)
    # Use functools.partial to pass processor to collate_fn
    from functools import partial
    collate_partial = partial(collate_fn, processor=processor)
    dataloader = DataLoader(pytorch_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_partial, num_workers=4, pin_memory=True)

    # Determine device
    if cfg.model.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.model.device)
    log.info(f"Using device: {device}")
    model.to(device)
    model.eval() # Ensure evaluation mode

    # Prepare HDF5 group and datasets
    split_group = hdf5_file.create_group(split_name)
    num_patches_per_image = 576 # Expected from Florence-2 VE output shape (577, dim) -> (576, dim)
    total_patches = len(indices) * num_patches_per_image
    feature_dim = model.config.vision_config.hidden_size # Should be 768 for base

    log.info(f"Preparing HDF5 dataset for {split_name}/encoded_patches with shape ({total_patches}, {feature_dim})")
    patch_ds = split_group.create_dataset(
        "encoded_patches",
        shape=(total_patches, feature_dim),
        dtype=np.float16,
        chunks=(cfg.batch_size * num_patches_per_image, feature_dim), # Adjust chunking as needed
        **hdf5plugin.Zstd(clevel=3)
    )

    if split_name == "validation": # Only save labels for the validation/analysis set
        log.info(f"Preparing HDF5 datasets for {split_name}/labels")
        labels_group = split_group.create_group("labels")
        label_ds = {}
        for label_name in ['age', 'gender', 'race']:
            label_ds[label_name] = labels_group.create_dataset(
                label_name,
                shape=(total_patches, 1),
                dtype=np.uint8, # Ensure uint8 for labels
                chunks=(cfg.batch_size * num_patches_per_image, 1),
                **hdf5plugin.Zstd(clevel=3)
            )

    # Process batches
    current_patch_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split_name} patches"):
            if batch is None: # Skip batch if collate_fn failed
                 continue
            processed_inputs = batch['processed_inputs'].to(device)
            batch_labels = batch['labels']
            batch_size = processed_inputs['pixel_values'].shape[0] # Actual batch size

            try:
                # --- Get Vision Encoder Output ---
                # This might differ depending on how src/deconstructed_florence.py works
                # Assuming direct model usage here. Adapt if you have a wrapper.
                vision_outputs = model.vision_tower(processed_inputs['pixel_values'], output_hidden_states=True)
                # Usually the last hidden state is what we need, before projection
                # Shape: (batch_size, sequence_length=577, hidden_size=768)
                image_features = vision_outputs.hidden_states[-1]
                # ----------------------------------

                # Extract patch embeddings (discard CLS token at index 0)
                # Shape: (batch_size, 576, 768)
                patch_embeddings = image_features[:, 1:, :]

                # Reshape for saving: (batch_size * 576, 768)
                patches_to_save = patch_embeddings.reshape(-1, feature_dim).cpu().numpy().astype(np.float16)

                # Write patches to HDF5
                n_patches_in_batch = patches_to_save.shape[0]
                patch_ds[current_patch_idx : current_patch_idx + n_patches_in_batch, :] = patches_to_save

                # Write labels if validation split
                if split_name == "validation":
                    for label_name in ['age', 'gender', 'race']:
                        # Repeat labels 576 times for each image in the batch
                        labels_to_save = batch_labels[label_name].unsqueeze(1).repeat(1, num_patches_per_image).reshape(-1, 1)
                        label_ds[label_name][current_patch_idx : current_patch_idx + n_patches_in_batch, :] = labels_to_save.numpy()

                current_patch_idx += n_patches_in_batch

            except Exception as e:
                log.error(f"Error processing batch during feature extraction: {e}")
                # Decide how to handle errors (e.g., skip batch, raise exception)
                # This example continues, potentially leaving gaps in HDF5

    log.info(f"Finished processing {split_name}. Total patches written: {current_patch_idx}/{total_patches}")
    if current_patch_idx != total_patches:
        log.warning(f"Mismatch in expected ({total_patches}) vs actual ({current_patch_idx}) patches written for {split_name}. Check for errors.")


# --- Global Model and Processor ---
# Load them once outside the main function to avoid reloading per task if running multiple tasks
# Note: This assumes the script is run once per invocation. If main.py calls this multiple times,
# you might need to pass model/processor or handle loading differently.
cfg_for_model_load = OmegaConf.load("config/model/florence2.yaml") # Load config directly for setup
log.info(f"Loading Florence-2 model: {cfg_for_model_load.model_identifier}")
model = AutoModelForCausalLM.from_pretrained(cfg_for_model_load.model_identifier, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(cfg_for_model_load.model_identifier, trust_remote_code=True)
# ----------------------------------

@hydra.main(config_path="../../config", config_name="task/extract_patches", version_base=None)
def main(cfg: DictConfig):
    log.info("Starting Florence-2 Patch Extraction...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_hdf5_path = cfg.path.patch_embeddings_file
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)

    log.info(f"Output HDF5 file: {output_hdf5_path}")

    try:
        with h5py.File(output_hdf5_path, 'w') as f: # Use 'w' to overwrite if exists
            f.attrs['description'] = 'Florence-2 patch embeddings from stratified FairFace subsets.'
            f.attrs['model_identifier'] = cfg.model.model_identifier
            f.attrs['stratify_columns'] = str(cfg.data.stratify_columns) # Store config info

            # Process Training Split
            extract_and_save_patches(cfg, "train", cfg.data.train_indices_file, f)

            # Process Validation Split
            extract_and_save_patches(cfg, "validation", cfg.data.val_indices_file, f)

        log.info("Patch extraction complete.")

    except Exception as e:
        log.error(f"An error occurred during patch extraction: {e}")
        # Clean up potentially corrupted HDF5 file?
        if os.path.exists(output_hdf5_path):
             log.warning(f"Removing potentially incomplete HDF5 file: {output_hdf5_path}")
             # os.remove(output_hdf5_path) # Uncomment carefully
        raise

if __name__ == "__main__":
    # Ensure hdf5plugin is loaded
    log.info(f"HDF5 Plugin version: {hdf5plugin.version.hdf5_plugin_version}")
    main() 