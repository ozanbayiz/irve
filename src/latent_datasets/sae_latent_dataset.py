import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# --- Constants ---
LATENT_DTYPE = np.float16
LABEL_DTYPE = np.uint8
DEFAULT_ZSTD_CLEVEL = 3 # Compression level (1-22)


def process_split(
    input_dset: h5py.Dataset,
    output_latent_dset: h5py.Dataset,
    n_samples: int,
    batch_size: int,
    model: torch.nn.Module,
    device: torch.device,
    latent_dtype: np.dtype,
) -> None:
    """
    Processes a single data split (e.g., training, validation) in batches.

    Reads encoded data, calculates the mean, encodes with the SAE,
    and writes the latent activations to the output HDF5 dataset.
    """
    log.info(
        f"Processing split with {n_samples} samples in batches of {batch_size}..."
    )
    model.eval() # Ensure model is in evaluation mode
    model_dtype = next(model.parameters()).dtype # Get model's expected dtype

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
            start_idx = i
            end_idx = min(i + batch_size, n_samples)

            # 1. Read Input Batch
            input_batch_np = input_dset[start_idx:end_idx]

            # 2. Calculate Mean
            # Ensure input is float32 for mean calculation stability if it's float16
            mean_batch_np = input_batch_np.astype(np.float32).mean(axis=1)

            # 3. Prepare for SAE
            mean_batch_torch = (
                torch.from_numpy(mean_batch_np).to(device).to(model_dtype)
            )

            # 4. Encode Batch
            # Assuming the model has an 'encode' method
            if hasattr(model, 'encode') and callable(model.encode):
                latent_batch_torch = model.encode(mean_batch_torch)
            else:
                # Fallback: assume forward pass does encoding
                # This might need adjustment based on the specific SAE model structure
                latent_batch_torch = model(mean_batch_torch)
                # If model outputs more than just latent (e.g., recon, loss), select latent
                if isinstance(latent_batch_torch, tuple):
                     # Assuming latent is the first element - ADJUST IF NEEDED
                    latent_batch_torch = latent_batch_torch[0]

            # 5. Prepare for Storage
            latent_batch_np = (
                latent_batch_torch.cpu().numpy().astype(latent_dtype)
            )

            # 6. Write Output Batch
            output_latent_dset[start_idx:end_idx] = latent_batch_np

    log.info(f"Finished processing split.")


def main(args):
    """Main function to generate the SAE encoded dataset."""

    input_path = Path(args.input_hdf5)
    output_path = Path(args.output_hdf5)
    checkpoint_path = Path(args.checkpoint)
    batch_size = args.batch_size

    # --- Input Validation ---
    if not input_path.is_file():
        log.error(f"Input HDF5 file not found: {input_path}")
        sys.exit(1)
    if not checkpoint_path.is_file():
        log.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Input HDF5: {input_path}")
    log.info(f"Output HDF5: {output_path}")
    log.info(f"SAE Checkpoint: {checkpoint_path}")
    log.info(f"Batch Size: {batch_size}")

    # --- Define Compression ---
    zstd_compression = hdf5plugin.Zstd(clevel=DEFAULT_ZSTD_CLEVEL)

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load SAE Model ---
    log.info("Loading SAE model from checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' not in checkpoint:
             log.error("Checkpoint does not contain 'config' key for Hydra instantiation.")
             sys.exit(1)
        if 'model_state_dict' not in checkpoint:
             log.error("Checkpoint does not contain 'model_state_dict' key.")
             sys.exit(1)

        cfg = OmegaConf.create(checkpoint['config'])
        # Ensure relative paths in config resolve correctly if needed,
        # though often model structure is self-contained in cfg.model
        # OmegaConf.resolve(cfg) # Uncomment if needed

        # Add hydra.utils to sys.path if necessary or ensure it's importable
        # This assumes your hydra config structure is standard
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Determine latent dimension (highly dependent on SAE model structure)
        # Attempt common ways to find the latent dimension - *ADJUST AS NEEDED*
        d_latent = None
        if hasattr(model, 'd_sae'):
            d_latent = model.d_sae
        elif hasattr(model, 'cfg') and hasattr(model.cfg, 'd_sae'):
             d_latent = model.cfg.d_sae
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'out_features'): # e.g. last linear layer
             d_latent = model.encoder.out_features
        elif hasattr(model, 'W_enc'): # Check for weight matrix convention
            d_latent = model.W_enc.shape[0]

        if d_latent is None:
            log.warning("Could not automatically determine latent dimension (D_latent).")
            # Attempt inference with a dummy batch to get output shape
            dummy_c_encoded = 1024 # Guess C_encoded if not available yet
            try:
                 with h5py.File(input_path, 'r') as f_in_dummy:
                      dummy_c_encoded = f_in_dummy['training']['encoded'].shape[2]
            except Exception:
                 log.warning(f"Could not read C_encoded from input file, using default {dummy_c_encoded}")

            log.info(f"Attempting dummy inference to find D_latent using C_encoded={dummy_c_encoded}...")
            dummy_input = torch.randn(2, dummy_c_encoded, device=device, dtype=next(model.parameters()).dtype)
            try:
                 with torch.no_grad():
                      dummy_output = model.encode(dummy_input) if hasattr(model, 'encode') else model(dummy_input)
                      if isinstance(dummy_output, tuple):
                           dummy_output = dummy_output[0] # Assume first output is latent
                      d_latent = dummy_output.shape[-1]
                      log.info(f"Inferred D_latent = {d_latent} from dummy output shape.")
            except Exception as e:
                 log.error(f"Dummy inference failed: {e}")
                 log.error("Cannot determine latent dimension. Please check model structure or set manually.")
                 sys.exit(1)
        else:
             log.info(f"Determined latent dimension D_latent = {d_latent}")

    except Exception as e:
        log.exception(f"Failed to load model: {e}")
        sys.exit(1)

    # --- Process Data ---
    try:
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            log.info("Input and output HDF5 files opened.")

            for split in ['training', 'validation']:
                log.info(f"--- Processing split: {split} ---")
                if split not in f_in:
                    log.warning(f"Split '{split}' not found in input file. Skipping.")
                    continue

                input_group = f_in[split]
                if 'encoded' not in input_group:
                    log.warning(f"'encoded' dataset not found in split '{split}'. Skipping.")
                    continue
                if 'labels' not in input_group:
                    log.warning(f"'labels' group not found in split '{split}'. Skipping label copy.")
                    has_labels = False
                else:
                    has_labels = True

                input_encoded_dset = input_group['encoded']
                n_samples = input_encoded_dset.shape[0]
                c_encoded = input_encoded_dset.shape[2] # Get C_encoded from actual data
                log.info(f"Found {n_samples} samples with C_encoded = {c_encoded}.")

                if n_samples == 0:
                    log.warning(f"Split '{split}' has 0 samples. Skipping processing.")
                    continue

                # Create output group
                output_group = f_out.create_group(split)

                # Pre-allocate latent activations dataset
                latent_chunk_shape = (min(batch_size, n_samples), d_latent)
                log.info(f"Creating output dataset '{split}/latent_activations' "
                         f"with shape=({n_samples}, {d_latent}), "
                         f"dtype={LATENT_DTYPE}, chunks={latent_chunk_shape}, compression=zstd")
                output_latent_dset = output_group.create_dataset(
                    'latent_activations',
                    shape=(n_samples, d_latent),
                    dtype=LATENT_DTYPE,
                    chunks=latent_chunk_shape,
                    compression=zstd_compression,
                )

                # Copy labels
                if has_labels:
                    log.info(f"Copying labels group for split '{split}'...")
                    f_out.copy(input_group['labels'], output_group, name='labels')
                    log.info("Labels copied.")
                else:
                    # Create an empty labels group maybe? Or just omit. Omitting for now.
                    log.warning(f"No labels group to copy for split '{split}'.")


                # Process the split in batches
                process_split(
                    input_dset=input_encoded_dset,
                    output_latent_dset=output_latent_dset,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    model=model,
                    device=device,
                    latent_dtype=LATENT_DTYPE,
                )

        log.info(f"Successfully created SAE encoded dataset: {output_path}")

    except Exception as e:
        log.exception(f"An error occurred during processing: {e}")
        # Clean up potentially partially written output file
        if output_path.exists():
            log.warning(f"Attempting to remove partially written file: {output_path}")
            try:
                os.remove(output_path)
            except OSError as rm_err:
                log.error(f"Failed to remove partial file: {rm_err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an HDF5 dataset with SAE latent activations from vision encoder outputs."
    )
    parser.add_argument(
        "--input-hdf5",
        type=str,
        required=True,
        help="Path to the input HDF5 file containing 'encoded' datasets.",
    )
    parser.add_argument(
        "--output-hdf5",
        type=str,
        required=True,
        help="Path for the output HDF5 file to be created.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the SAE model checkpoint file (.pth).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing.",
    )

    args = parser.parse_args()
    main(args) 