# REMOVE hydra import if only used for the decorator
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from datasets import load_dataset
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def create_stratified_subset(dataset, stratify_columns, target_samples_per_stratum, seed):
    """Creates a stratified subset of a Hugging Face dataset."""
    log.info(f"Creating stratified subset based on columns: {stratify_columns}")
    df = dataset.to_pandas()

    if not all(col in df.columns for col in stratify_columns):
        raise ValueError(f"One or more stratify_columns ({stratify_columns}) not found in dataset columns ({df.columns})")

    grouped = df.groupby(stratify_columns)
    sampled_indices = []
    actual_samples_per_stratum = {}

    log.info(f"Found {len(grouped)} strata.")

    for name, group in grouped:
        n_available = len(group)
        n_to_sample = min(target_samples_per_stratum, n_available)
        actual_samples_per_stratum[str(name)] = n_to_sample

        if n_to_sample > 0:
            sampled_group_indices = group.sample(n=n_to_sample, random_state=seed, replace=False).index
            sampled_indices.extend(sampled_group_indices)
        else:
            log.warning(f"Stratum {name} has 0 samples available or target is 0.")

    log.info(f"Total indices sampled before shuffling: {len(sampled_indices)}")

    # Log actual counts per stratum
    log.info("Actual samples per stratum:")
    for stratum, count in actual_samples_per_stratum.items():
        log.info(f"  - {stratum}: {count}")

    # Shuffle the final list of indices
    rng = np.random.RandomState(seed)
    final_indices = rng.permutation(sampled_indices).tolist()

    log.info(f"Total indices sampled after shuffling: {len(final_indices)}")

    # Return the list of indices for selection
    return final_indices, actual_samples_per_stratum

# REMOVE the @hydra.main decorator
# @hydra.main(config_path="../../config", config_name="task/prepare_subset", version_base=None)
# RENAME the function slightly to avoid confusion with the outer main.py, or keep as main
# It now accepts the fully resolved config 'cfg' from the caller (src/main.py)
def run_task(cfg: DictConfig):
    log.info("Starting FairFace Stratified Subsetting Task...")
    # Config is already resolved, use it directly
    log.info(f"Using Configuration passed from main runner:\n{OmegaConf.to_yaml(cfg)}")

    seed = cfg.data.random_seed
    stratify_columns = list(cfg.data.stratify_columns) # Ensure it's a list

    # --- Load Full Dataset ---
    try:
        log.info("Loading full FairFace dataset from Hugging Face...")
        # Adjust split names if necessary (e.g., 'train', 'validation')
        full_train_ds = load_dataset("huggingface/fairface", split="train")
        full_val_ds = load_dataset("huggingface/fairface", split="validation")
        log.info(f"Loaded train split with {len(full_train_ds)} samples.")
        log.info(f"Loaded validation split with {len(full_val_ds)} samples.")
    except Exception as e:
        log.error(f"Failed to load FairFace dataset: {e}")
        raise

    # --- Create Subsets ---
    log.info("Creating training subset...")
    train_indices, train_stratum_counts = create_stratified_subset(
        full_train_ds,
        stratify_columns,
        cfg.data.target_samples_per_stratum_train,
        seed
    )

    log.info("Creating validation subset...")
    val_indices, val_stratum_counts = create_stratified_subset(
        full_val_ds,
        stratify_columns,
        cfg.data.target_samples_per_stratum_val,
        seed + 1 # Use a different seed for validation sampling for robustness
    )

    # --- Save Indices ---
    # Access paths directly from the passed cfg object
    output_dir = cfg.data.subset_indices_dir
    train_output_file = cfg.data.train_indices_file
    val_output_file = cfg.data.val_indices_file

    try:
        # Use os.path.join for better path construction
        full_train_path = os.path.join(hydra.utils.get_original_cwd(), train_output_file)
        full_val_path = os.path.join(hydra.utils.get_original_cwd(), val_output_file)
        full_output_dir = os.path.dirname(full_train_path) # Get dir from full path

        os.makedirs(full_output_dir, exist_ok=True)
        log.info(f"Saving train indices to: {full_train_path}")
        np.save(full_train_path, np.array(train_indices, dtype=int))

        log.info(f"Saving validation indices to: {full_val_path}")
        np.save(full_val_path, np.array(val_indices, dtype=int))

    except Exception as e:
        log.error(f"Failed to save subset indices: {e}")
        raise

    log.info("Stratified subset index generation complete.")
    log.info(f"Train subset size: {len(train_indices)}")
    log.info(f"Validation subset size: {len(val_indices)}")


# REMOVE the __main__ block as this script is not meant to be run directly anymore
# if __name__ == "__main__":
#     main() 