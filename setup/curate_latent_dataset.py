import torch
import h5py
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import sys
import os
from typing import Dict, List, Tuple, Optional
from functools import partial # For cleaner collate_fn binding

# Set environment variable to suppress the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
try:
    from src.deconstructed_florence import DeconstructedFlorence2, FlorenceVisionEncoder
    from transformers import AutoProcessor
    from datasets import load_dataset, Dataset, Features, Value, Image as HFImage, ClassLabel
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print(f"Attempted to import from: {src_dir}", file=sys.stderr)
    print("Please ensure 'src' directory with 'deconstructed_florence.py' is accessible.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
DATASET_NAME = 'HuggingFaceM4/FairFace'
DATASET_CONFIG = '0.25'
IMAGE_COLUMN = 'image'
LABEL_COLUMNS = ['age', 'gender', 'race'] # Specific labels for FairFace
MODEL_ID = "microsoft/Florence-2-base"
HDF5_OUTPUT_PATH = "output_data/fairface_latent_dataset.hdf5" # Specific output name
BATCH_SIZE = 64          # Adjust based on GPU memory
FORCE_CPU = False        # Set to True to force CPU usage
NUM_WORKERS = 4          # Number of DataLoader workers
COMPRESSION_TYPE = "gzip" # Compression for HDF5

# --- Setup Device and Dtype ---
if FORCE_CPU:
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Compute Dtype (Model runs with this)
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
# Storage Dtypes (HDF5 stores these)
FLOAT_STORAGE_DTYPE = np.float16
LABEL_STORAGE_DTYPE = np.uint8 # Assuming < 256 classes for age, gender, race in FairFace
print(f"Using device: {DEVICE}, compute dtype: {COMPUTE_DTYPE}")
print(f"Storage dtypes: float={FLOAT_STORAGE_DTYPE}, label={LABEL_STORAGE_DTYPE}")
PIN_MEMORY = (DEVICE == 'cuda')

# --- Helper Functions ---

def check_columns(dataset_features: Features, image_col: str, label_cols: List[str]):
    """Validate required image and label columns exist."""
    if image_col not in dataset_features:
        raise ValueError(f"Image column '{image_col}' not found in dataset features: {list(dataset_features.keys())}")
    if not isinstance(dataset_features[image_col], HFImage):
         print(f"Warning: Image column '{image_col}' is not of type datasets.Image. Ensure it contains loadable PIL images.", file=sys.stderr)
    for label_col in label_cols:
        if label_col not in dataset_features:
            raise ValueError(f"Label column '{label_col}' not found in dataset features: {list(dataset_features.keys())}")
        # FairFace uses ClassLabel, which is expected
        if not isinstance(dataset_features[label_col], ClassLabel):
             print(f"Warning: Label column '{label_col}' is not of type datasets.ClassLabel. Will cast to {LABEL_STORAGE_DTYPE}.", file=sys.stderr)

def get_shapes_and_dtypes(
    processor: AutoProcessor,
    vision_encoder: FlorenceVisionEncoder,
    sample_image: Image.Image,
    label_columns: List[str],
    device: str,
    compute_dtype: torch.dtype
) -> Tuple[Tuple, np.dtype, Tuple, np.dtype, Dict[str, np.dtype]]:
    """Process one sample to determine shapes and storage dtypes for HDF5."""
    dummy_text = "<DUMMY>" # Placeholder text for processor
    inputs = processor(text=dummy_text, images=sample_image, return_tensors="pt")
    pixel_values_sample = inputs['pixel_values'].to(device=device, dtype=compute_dtype)

    with torch.no_grad():
        # Use encode() to get final encoded features, not intermediate latents
        encoded_features_sample = vision_encoder(pixel_values_sample)

    raw_shape = pixel_values_sample.shape # Includes batch dim 1
    encoded_shape = encoded_features_sample.shape # Includes batch dim 1

    # Determine storage dtypes (float16 for continuous, uint8 for labels)
    raw_storage_dtype = FLOAT_STORAGE_DTYPE
    encoded_storage_dtype = FLOAT_STORAGE_DTYPE
    label_storage_dtypes = {label_col: LABEL_STORAGE_DTYPE for label_col in label_columns}

    # Return shapes *without* the sample batch dimension
    raw_shape_tpl = tuple(raw_shape[1:])
    encoded_shape_tpl = tuple(encoded_shape[1:])

    return raw_shape_tpl, raw_storage_dtype, encoded_shape_tpl, encoded_storage_dtype, label_storage_dtypes

# Define the custom collate function for FairFace
def collate_batch(batch_list: List[Dict], processor: AutoProcessor, image_col: str, label_cols: List[str]) -> Dict:
    """
    Collates FairFace samples: processes images and packages multiple labels.
    """
    images = [item[image_col] for item in batch_list]
    # Create a dictionary of label lists
    labels_dict = {label_col: [item[label_col] for item in batch_list] for label_col in label_cols}

    dummy_texts = ["<IGNORE>" for _ in images]

    try:
        inputs = processor(text=dummy_texts, images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']
    except Exception as e:
        print(f"\nError in collate_fn processing images: {e}", file=sys.stderr)
        return {'pixel_values': torch.empty(0), 'labels': {label_col: torch.empty(0, dtype=torch.long) for label_col in label_cols}}

    # Convert each label list to a tensor
    labels_tensor_dict = {
        label_col: torch.tensor(labels, dtype=torch.long)
        for label_col, labels in labels_dict.items()
    }

    return {'pixel_values': pixel_values, 'labels': labels_tensor_dict}


def process_and_store_split(
    dataset_split: Dataset,
    hdf5_group: h5py.Group,
    processor: AutoProcessor,
    vision_encoder: FlorenceVisionEncoder,
    image_col: str,
    label_cols: List[str], # Expect list of label columns
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: str,
    compute_dtype: torch.dtype,
    split_name: str
):
    """Processes a dataset split and stores results in the HDF5 group (optimized)."""
    print(f"\nProcessing split: {split_name}")
    current_offset = 0

    # Access HDF5 datasets
    raw_dset = hdf5_group['raw']
    encoded_dset = hdf5_group['encoded']
    label_dsets = {label_col: hdf5_group[f'labels/{label_col}'] for label_col in label_cols}

    # --- Batch Processing ---
    # Use functools.partial for cleaner binding of arguments to collate_fn
    collate_fn_partial = partial(collate_batch, processor=processor, image_col=image_col, label_cols=label_cols)

    dataloader = torch.utils.data.DataLoader(
        dataset_split,
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for batch in tqdm(dataloader, desc=f"Processing {split_name}", total=len(dataloader)):
        pixel_values = batch['pixel_values']
        labels_batch_dict = batch['labels'] # Now a dictionary of label tensors

        pixel_values = pixel_values.to(device=device, dtype=compute_dtype, non_blocking=pin_memory)

        current_batch_size = pixel_values.shape[0]
        if current_batch_size == 0:
             continue

        with torch.no_grad():
            # Get only the final encoded features
            encoded_features = vision_encoder(pixel_values)

        # Write data directly to the pre-allocated slice
        end_offset = current_offset + current_batch_size

        # Cast to float16 for storage
        raw_dset[current_offset:end_offset] = pixel_values.cpu().to(torch.float16).numpy()
        encoded_dset[current_offset:end_offset] = encoded_features.cpu().to(torch.float16).numpy()

        # Write each label type, casting to uint8
        for label_name, label_tensor in labels_batch_dict.items():
            label_dset = label_dsets[label_name]
            label_dset[current_offset:end_offset] = label_tensor.cpu().numpy().astype(LABEL_STORAGE_DTYPE)

        current_offset = end_offset

    print(f"Finished processing {split_name}. Total items processed: {current_offset}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting FairFace dataset curation (Memory Optimized)...")

    # 1. Load Dataset & Get Sizes
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    except Exception as e:
        print(f"Error loading dataset '{DATASET_NAME}': {e}", file=sys.stderr)
        sys.exit(1)

    # FairFace has 'train' and 'validation' splits
    if 'train' not in dataset or 'validation' not in dataset:
        raise ValueError(f"Dataset '{DATASET_NAME}' must contain 'train' and 'validation' splits.")
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    N_train = len(train_dataset)
    N_val = len(val_dataset)
    print(f"Train samples: {N_train}, Validation samples: {N_val}")

    # Validate columns
    try:
         check_columns(train_dataset.features, IMAGE_COLUMN, LABEL_COLUMNS)
         check_columns(val_dataset.features, IMAGE_COLUMN, LABEL_COLUMNS)
    except ValueError as e:
         print(e, file=sys.stderr)
         sys.exit(1)

    # 2. Load Model and Processor
    print(f"Loading model: {MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        deconstructed_model = DeconstructedFlorence2(MODEL_ID, device=DEVICE, dtype=COMPUTE_DTYPE, trust_remote_code=True)
        deconstructed_model.model.eval()
        vision_encoder = deconstructed_model.vision_encoder
    except Exception as e:
        print(f"Error loading model or processor '{MODEL_ID}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Determine Shapes and Dtypes
    print("Determining HDF5 shapes and dtypes...")
    try:
        sample_data = next(iter(train_dataset))
        sample_image = sample_data[IMAGE_COLUMN]
        if not isinstance(sample_image, Image.Image):
             print(f"Warning: Sample image is not PIL Image, type is {type(sample_image)}. Attempting to load.")
             try:
                 if isinstance(sample_image, dict) and 'bytes' in sample_image and sample_image['bytes']:
                     from io import BytesIO
                     sample_image = Image.open(BytesIO(sample_image['bytes'])).convert("RGB")
                 elif isinstance(sample_image, str):
                      sample_image = Image.open(sample_image).convert("RGB")
                 else:
                     raise ValueError("Cannot automatically load sample image.")
             except Exception as load_err:
                  print(f"Failed to load sample image: {load_err}", file=sys.stderr)
                  sys.exit(1)

        # Pass label columns list
        raw_shape_tpl, raw_storage_dtype, encoded_shape_tpl, encoded_storage_dtype, label_storage_dtypes = get_shapes_and_dtypes(
            processor, vision_encoder, sample_image, LABEL_COLUMNS, DEVICE, COMPUTE_DTYPE
        )
        print(f"  Raw pixel shape (C, H, W): {raw_shape_tpl}, storage dtype: {raw_storage_dtype}")
        print(f"  Encoded shape (N, C): {encoded_shape_tpl}, storage dtype: {encoded_storage_dtype}")
        for label_col, dtype in label_storage_dtypes.items():
             print(f"  Label '{label_col}' storage dtype: {dtype}")

    except Exception as e:
        print(f"Error determining shapes/dtypes: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Initialize HDF5 File (Optimized - Pre-allocate)
    print(f"Initializing HDF5 file (pre-allocated): {HDF5_OUTPUT_PATH}")
    os.makedirs(os.path.dirname(HDF5_OUTPUT_PATH), exist_ok=True)
    try:
        with h5py.File(HDF5_OUTPUT_PATH, 'w') as file:
            train_group = file.create_group('training')
            val_group = file.create_group('validation')

            for group, n_samples, split_name in [(train_group, N_train, 'train'), (val_group, N_val, 'validation')]:
                if n_samples == 0: continue
                print(f"  Creating datasets for '{split_name}' group (size: {n_samples})...")

                # Raw data
                raw_chunks = (min(BATCH_SIZE*2, n_samples),) + raw_shape_tpl
                group.create_dataset('raw', shape=(n_samples,) + raw_shape_tpl, dtype=raw_storage_dtype, chunks=raw_chunks, compression=COMPRESSION_TYPE)

                # Encoded data
                encoded_chunks = (min(BATCH_SIZE, n_samples),) + encoded_shape_tpl
                group.create_dataset('encoded', shape=(n_samples,) + encoded_shape_tpl, dtype=encoded_storage_dtype, chunks=encoded_chunks, compression=COMPRESSION_TYPE)

                # Labels group
                labels_group = group.create_group('labels')
                label_chunks = (min(BATCH_SIZE * 10, n_samples),)
                for label_col, label_dtype in label_storage_dtypes.items():
                     labels_group.create_dataset(label_col, shape=(n_samples,), dtype=label_dtype, chunks=label_chunks, compression=COMPRESSION_TYPE)

            # 5. & 6. Process and Store Splits (within the same 'with' block)
            print("\nStarting data processing and storage...")

            if N_train > 0:
                process_and_store_split(
                    train_dataset, train_group, processor, vision_encoder,
                    IMAGE_COLUMN, LABEL_COLUMNS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
                    DEVICE, COMPUTE_DTYPE, 'training' # Removed num_blocks
                )
            else:
                print("Skipping processing for empty 'training' split.")

            if N_val > 0:
                process_and_store_split(
                    val_dataset, val_group, processor, vision_encoder,
                    IMAGE_COLUMN, LABEL_COLUMNS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
                    DEVICE, COMPUTE_DTYPE, 'validation' # Removed num_blocks
                )
            else:
                 print("Skipping processing for empty 'validation' split.")

    except Exception as e:
        print(f"\nError during HDF5 creation or data processing/storage: {e}", file=sys.stderr)
        sys.exit(1)

    # 7. Cleanup (Handled by 'with' statement)
    print(f"\nDataset curation complete. Output saved to: {HDF5_OUTPUT_PATH}")
