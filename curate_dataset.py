from datasets import load_dataset
from IPython.display import display
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import requests
import torch
import gc  # Import garbage collector
import h5py # Import h5py
import numpy as np # Import numpy
import os # Add os import for path checking in verification
import math # Add math import for ceiling division

# --- Configuration ---
MODEL_NAME = "microsoft/Florence-2-base"
DATASET_NAME = 'HuggingFaceM4/FairFace'
DATASET_CONFIG = '1.25'
NUM_TRAIN_SAMPLES = 4000  # Max number of *training* images to process
# Process all validation samples by default
BATCH_SIZE = 16
PREFERRED_DEVICE = "cuda:0"
HDF5_SAVE_PATH = "collected_data.h5" # Renamed for clarity
TASK_PROMPT = "<OD>"
LABEL_KEYS = ["race", "gender", "age"] # Labels to extract from FairFace

# --- Helper Functions ---

def setup_environment(preferred_device):
    """Sets up the device and dtype for PyTorch."""
    try:
        device = torch.device(preferred_device if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() and "cuda" in str(device) else torch.float32
        print(f"Using device: {device}, dtype: {torch_dtype}")
        return device, torch_dtype
    except Exception as e:
        print(f"Error setting up device or dtype: {e}")
        print("Falling back to CPU and float32.")
        return torch.device("cpu"), torch.float32

def load_data(dataset_name, dataset_config):
    """Loads the train and validation splits of the specified dataset."""
    datasets = {}
    try:
        datasets['train'] = load_dataset(dataset_name, dataset_config, split='train')
        print(f"Loaded train dataset: {dataset_name} ({dataset_config})")
    except Exception as e:
        print(f"Error loading train dataset: {e}")
        # Decide if we should exit or continue without train data
        exit()
    try:
        datasets['validation'] = load_dataset(dataset_name, dataset_config, split='validation')
        print(f"Loaded validation dataset: {dataset_name} ({dataset_config})")
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        # Decide if we should exit or continue without validation data
        exit()
    return datasets

def load_model_and_processor(model_name, device, torch_dtype):
    """Loads the model and processor."""
    try:
        # Load model onto the specified device directly
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        # Processor doesn't need device placement usually
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loaded model and processor: {model_name}")
        return model, processor
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        exit()

activation_dict = {}
hook_handles = []

def get_activation_hook(name):
    """Creates a hook function to capture activations."""
    def hook(model, input, output):
        activation_output = output[0] if isinstance(output, tuple) else output
        activation_dict[name] = activation_output.detach().cpu()
    return hook

def register_activation_hooks(model):
    """Registers forward hooks to capture activations from specified layers."""
    global hook_handles
    hook_handles = []
    hook_names = []
    try:
        main_hook_name = 'DaViT'
        handle = model.vision_tower.register_forward_hook(get_activation_hook(main_hook_name))
        hook_handles.append(handle)
        hook_names.append(main_hook_name)
        print(f"Registered hook for: {main_hook_name}")

        for i, block in enumerate(model.vision_tower.blocks):
            block_hook_name = f'DaViT_block_{i}'
            handle = block.register_forward_hook(get_activation_hook(block_hook_name))
            hook_handles.append(handle)
            hook_names.append(block_hook_name)
            print(f"Registered hook for: {block_hook_name}")
        return hook_names
    except AttributeError as e:
        print(f"Error registering hooks: {e}. Model structure might be different than expected.")
        remove_hooks()
        exit()

def remove_hooks():
    """Removes all registered hooks."""
    global hook_handles
    if not hook_handles: return # Avoid printing if no hooks were registered
    print("\nRemoving hooks...")
    for handle in hook_handles:
        handle.remove()
    hook_handles = []
    print("Hooks removed.")

def get_activation_shapes(model, processor, dataset, device, torch_dtype, hook_names):
    """Runs a single forward pass on the first valid image of the *train* dataset."""
    global activation_dict
    activation_shapes = {}
    first_valid_image = None
    print("Searching for the first valid image in the training dataset...")
    for i in range(len(dataset)):
        if dataset[i]['image'] is not None:
            first_valid_image = dataset[i]['image']
            print(f"Found valid image at index {i}.")
            break

    if first_valid_image is None:
        print("Error: No valid images found in the training dataset to determine shapes.")
        return None

    print("Determining activation shapes...")
    # Ensure hooks are registered before running this
    if not hook_handles:
        print("Error: Hooks must be registered before determining shapes.")
        return None

    try:
        inputs = processor(text=TASK_PROMPT, images=first_valid_image, return_tensors="pt").to(device, torch_dtype)
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1, do_sample=False, num_beams=1,
            )

        if not activation_dict:
             print("Warning: No activations captured during shape determination run. Check hook registration.")
             return None

        for name in hook_names: # Use hook_names to ensure order and completeness
             if name in activation_dict:
                activation_tensor = activation_dict[name]
                # Shape excludes the batch dimension (which is 1 here)
                activation_shapes[name] = tuple(activation_tensor.shape[1:])
                print(f"Shape for '{name}': {activation_shapes[name]}")
             else:
                 print(f"Warning: Activation for '{name}' not captured during shape determination.")
                 # Decide how to handle this - return None or empty shape? Let's return None.
                 activation_dict.clear()
                 return None


        activation_dict.clear()
        del inputs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        return activation_shapes

    except Exception as e:
        print(f"Error during shape determination: {e}")
        activation_dict.clear()
        return None

def initialize_hdf5_file(save_path, datasets, num_train_samples_config, activation_shapes, hook_names, label_keys):
    """Creates HDF5 file and pre-allocates datasets for train/validation splits."""
    print(f"\nInitializing HDF5 file at {save_path}...")
    valid_samples_count = {'train': 0, 'validation': 0}
    label_info = {} # Store label features if needed for dtype/shape

    # Count valid samples and get label info from train split
    train_dataset = datasets.get('train')
    if train_dataset:
        num_to_check = min(num_train_samples_config, len(train_dataset))
        print(f"Counting valid training samples (up to {num_to_check})...")
        for i in range(num_to_check):
            if train_dataset[i]['image'] is not None:
                valid_samples_count['train'] += 1
        # Get label info (assuming consistent across splits)
        for key in label_keys:
            if key in train_dataset.features:
                 # Fairface uses ClassLabel -> int64
                 label_info[key] = {'dtype': np.int64} # Use int64 based on datasets feature type
            else:
                 print(f"Warning: Label key '{key}' not found in training dataset features.")


    val_dataset = datasets.get('validation')
    if val_dataset:
        num_to_check = len(val_dataset) # Check all validation samples
        print(f"Counting valid validation samples (up to {num_to_check})...")
        for i in range(num_to_check):
            if val_dataset[i]['image'] is not None:
                valid_samples_count['validation'] += 1

    print(f"Expected valid samples: Train={valid_samples_count['train']}, Validation={valid_samples_count['validation']}")

    try:
        hf = h5py.File(save_path, 'w')

        for split in ['train', 'validation']:
            if valid_samples_count[split] == 0:
                print(f"Skipping dataset creation for '{split}' split (0 valid samples).")
                continue

            split_group = hf.create_group(split)
            act_group = split_group.create_group('activations')
            lbl_group = split_group.create_group('labels')
            num_valid = valid_samples_count[split]

            # Create activation datasets
            for name in hook_names:
                 if name in activation_shapes:
                    shape = activation_shapes[name]
                    dataset_shape = (num_valid,) + shape
                    # Choose dtype based on model's torch_dtype? Let's stick to float32 for broader compatibility.
                    act_group.create_dataset(name, shape=dataset_shape, dtype=np.float32, chunks=True)
                    print(f"  Created {split}/activations/{name} dataset with shape {dataset_shape}")
                 else:
                     print(f"Warning: Shape for activation '{name}' not found. Cannot create dataset.")


            # Create label datasets
            for key in label_keys:
                if key in label_info:
                    dtype = label_info[key]['dtype']
                    lbl_group.create_dataset(key, shape=(num_valid,), dtype=dtype, chunks=True)
                    print(f"  Created {split}/labels/{key} dataset with shape {(num_valid,)}")

        return hf, valid_samples_count
    except Exception as e:
        print(f"Error initializing HDF5 file {save_path}: {e}")
        if 'hf' in locals() and hf: hf.close() # Close if partially open
        return None, {}


def extract_and_write_split(split_name, model, processor, dataset, num_samples_to_process, batch_size, task_prompt, device, torch_dtype, hook_names, label_keys, hf):
    """Runs inference on a specific split and writes activations/labels directly to HDF5."""
    global activation_dict
    effective_num_samples = min(num_samples_to_process, len(dataset))
    if effective_num_samples == 0:
        print(f"\nNo samples to process for '{split_name}' split.")
        return

    # Check if groups exist (might have been skipped if 0 valid samples initially counted)
    if split_name not in hf:
        print(f"\nHDF5 group for '{split_name}' not found. Skipping processing.")
        return
    if 'activations' not in hf[split_name] or 'labels' not in hf[split_name]:
        print(f"\nHDF5 subgroups 'activations' or 'labels' not found for '{split_name}'. Skipping processing.")
        return

    act_group = hf[split_name]['activations']
    lbl_group = hf[split_name]['labels']
    write_cursor = 0 # Tracks the index for the next write in HDF5 for this split

    print(f"\nStarting extraction for '{split_name}' split ({effective_num_samples} samples, batch size {batch_size})...")

    try:
        for i in range(0, effective_num_samples, batch_size):
            batch_start_idx = i
            batch_end_idx = min(i + batch_size, effective_num_samples)

            # Prepare batch
            image_batch = []
            label_batch = {key: [] for key in label_keys}
            valid_batch_indices = [] # Track indices relative to the start of the batch (0 to batch_size-1)

            for idx_in_dataset in range(batch_start_idx, batch_end_idx):
                item = dataset[idx_in_dataset]
                if item['image'] is not None:
                    image_batch.append(item['image'])
                    valid_batch_indices.append(idx_in_dataset - batch_start_idx)
                    for key in label_keys:
                        if key in item:
                            label_batch[key].append(item[key])
                        else:
                            # Handle missing label key - append a default? Or error?
                            print(f"Warning: Label key '{key}' missing for sample {idx_in_dataset}. Appending -1.")
                            label_batch[key].append(-1) # Use -1 as placeholder
                # else: # Reduce verbosity
                    # print(f"Skipping sample {idx_in_dataset} (image is None)")


            if not image_batch: # Skip if batch is empty
                 # print(f"Skipping batch {batch_start_idx}-{batch_end_idx} (all images None)")
                 continue

            actual_current_batch_size = len(image_batch)

            try:
                # Process the batch
                inputs = processor(text=[task_prompt] * actual_current_batch_size, images=image_batch, return_tensors="pt").to(device, torch_dtype)

                with torch.no_grad():
                    _ = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1, do_sample=False, num_beams=1,
                    )

                # Write captured activations and labels for this batch to HDF5
                batch_end_cursor = write_cursor + actual_current_batch_size
                if batch_end_cursor > act_group[hook_names[0]].shape[0]: # Basic check
                     print(f"Error: Write cursor ({batch_end_cursor}) exceeds dataset size for {split_name}. Stopping.")
                     break


                # Write Activations
                for name, batch_activation_tensor in activation_dict.items():
                    if name in act_group:
                        try:
                            numpy_batch = batch_activation_tensor.numpy().astype(np.float32) # Already on CPU
                            act_group[name][write_cursor : batch_end_cursor] = numpy_batch
                        except Exception as write_e:
                            print(f"Error writing batch for {split_name}/activations/{name} (indices {write_cursor}:{batch_end_cursor}): {write_e}")
                    # else: # Warning printed during init
                        # print(f"Warning: HDF5 dataset {split_name}/activations/{name} not found.")

                # Write Labels
                for key in label_keys:
                    if key in lbl_group:
                         try:
                             # Convert list of labels to numpy array with correct dtype
                             dtype = lbl_group[key].dtype
                             numpy_labels = np.array(label_batch[key], dtype=dtype)
                             lbl_group[key][write_cursor : batch_end_cursor] = numpy_labels
                         except Exception as write_e:
                            print(f"Error writing batch for {split_name}/labels/{key} (indices {write_cursor}:{batch_end_cursor}): {write_e}")
                    # else: # Warning printed during init
                         # print(f"Warning: HDF5 dataset {split_name}/labels/{key} not found.")


                activation_dict.clear()
                del inputs, image_batch, label_batch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()

                print(f"  {split_name}: Processed samples {batch_start_idx + 1}-{batch_end_idx}. Wrote {actual_current_batch_size} items. (Cursor: {batch_end_cursor})")
                write_cursor = batch_end_cursor

            except Exception as e:
                print(f"Error processing batch for '{split_name}' starting at sample {batch_start_idx}: {e}")
                activation_dict.clear()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
                continue # Continue to the next batch

    except Exception as outer_e:
         print(f"An error occurred during processing of '{split_name}' split: {outer_e}")
    finally:
        # Hooks are removed once after processing all splits in main
        pass

    print(f"\nFinished processing '{split_name}'. Total valid items written: {write_cursor}")


def verify_saved_activations(save_path, expected_counts):
    """Verifies the saved HDF5 file structure and dataset shapes."""
    print(f"\nVerification: Loading saved data from {save_path}...")
    if not os.path.exists(save_path):
        print(f"File not found for verification: {save_path}")
        return

    try:
        with h5py.File(save_path, 'r') as hf:
            print(f"Successfully opened {save_path}")
            print("Top-level groups:", list(hf.keys()))

            for split in ['train', 'validation']:
                if split not in hf:
                    print(f"Group '{split}' not found.")
                    continue
                print(f"\nVerifying '{split}' group:")
                split_group = hf[split]
                print("  Subgroups:", list(split_group.keys()))

                if 'activations' in split_group:
                    act_group = split_group['activations']
                    print("  Activations datasets:", list(act_group.keys()))
                    for name in act_group.keys():
                        dset = act_group[name]
                        print(f"    '{name}': Shape={dset.shape}, Dtype={dset.dtype}")
                        if dset.shape[0] != expected_counts.get(split, -1):
                             print(f"      Warning: Expected {expected_counts.get(split)} samples, found {dset.shape[0]}")

                if 'labels' in split_group:
                    lbl_group = split_group['labels']
                    print("  Labels datasets:", list(lbl_group.keys()))
                    for name in lbl_group.keys():
                        dset = lbl_group[name]
                        print(f"    '{name}': Shape={dset.shape}, Dtype={dset.dtype}")
                        if dset.shape[0] != expected_counts.get(split, -1):
                             print(f"      Warning: Expected {expected_counts.get(split)} samples, found {dset.shape[0]}")

    except Exception as e:
        print(f"Error during HDF5 verification load: {e}")


# --- Main Execution ---

def main():
    """Main function to orchestrate the data curation process."""
    # Setup
    device, torch_dtype = setup_environment(PREFERRED_DEVICE)

    # Load Data (Train and Validation)
    datasets = load_data(DATASET_NAME, DATASET_CONFIG)
    if not datasets:
        print("Failed to load datasets. Exiting.")
        return

    # Load Model
    model, processor = load_model_and_processor(MODEL_NAME, device, torch_dtype)

    # Register Hooks
    hook_names = register_activation_hooks(model)
    if not hook_names:
        print("Failed to register hooks. Exiting.")
        return

    # Determine Activation Shapes (using train data)
    train_dataset = datasets.get('train')
    if not train_dataset:
         print("Training data not available for shape determination. Exiting.")
         remove_hooks()
         return
    activation_shapes = get_activation_shapes(model, processor, train_dataset, device, torch_dtype, hook_names)
    if activation_shapes is None:
        print("Failed to determine activation shapes. Exiting.")
        remove_hooks()
        return

    # Initialize HDF5 File and Datasets
    hf, valid_samples_counts = initialize_hdf5_file(HDF5_SAVE_PATH, datasets, NUM_TRAIN_SAMPLES, activation_shapes, hook_names, LABEL_KEYS)
    if hf is None:
        print("Failed to initialize HDF5 file. Exiting.")
        remove_hooks()
        return

    # Extract Activations and Write Incrementally (for each split)
    hf_closed = False
    try:
        if 'train' in datasets and valid_samples_counts.get('train', 0) > 0:
             num_train_to_process = min(NUM_TRAIN_SAMPLES, len(datasets['train']))
             extract_and_write_split(
                 'train', model, processor, datasets['train'], num_train_to_process, BATCH_SIZE, TASK_PROMPT, device, torch_dtype, hook_names, LABEL_KEYS, hf
             )
        else:
            print("\nSkipping training data processing.")

        if 'validation' in datasets and valid_samples_counts.get('validation', 0) > 0:
             num_val_to_process = len(datasets['validation']) # Process all validation
             extract_and_write_split(
                 'validation', model, processor, datasets['validation'], num_val_to_process, BATCH_SIZE, TASK_PROMPT, device, torch_dtype, hook_names, LABEL_KEYS, hf
             )
        else:
            print("\nSkipping validation data processing.")

    except Exception as e:
         print(f"\nAn unexpected error occurred during split processing: {e}")
    finally:
        # Ensure hooks are removed AFTER processing ALL splits
        remove_hooks()
        # Ensure HDF5 file is closed
        if hf and not hf_closed:
            print("\nClosing HDF5 file...")
            try:
                hf.close()
                hf_closed = True
                print("HDF5 file closed.")
            except Exception as close_e:
                print(f"Error closing HDF5 file: {close_e}")

    # Optional: Clear model and processor from memory
    print("\nClearing model and processor from memory...")
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model and processor cleared.")

    # Verify Saved Data (Reads the closed file)
    if os.path.exists(HDF5_SAVE_PATH):
        verify_saved_activations(HDF5_SAVE_PATH, valid_samples_counts)
    else:
        print(f"\nSkipping verification: File {HDF5_SAVE_PATH} does not exist.")


    print("\nScript finished.")

if __name__ == "__main__":
    main()