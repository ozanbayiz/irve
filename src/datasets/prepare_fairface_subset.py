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


@hydra.main(config_path="../../config", config_name="task/prepare_subset", version_base=None)
def main(cfg: DictConfig):
    log.info("Starting FairFace Stratified Subsetting...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

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
    output_dir = cfg.data.subset_indices_dir
    train_output_file = cfg.data.train_indices_file
    val_output_file = cfg.data.val_indices_file

    try:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving train indices to: {train_output_file}")
        np.save(train_output_file, np.array(train_indices, dtype=int))

        log.info(f"Saving validation indices to: {val_output_file}")
        np.save(val_output_file, np.array(val_indices, dtype=int))

        # Optionally save stratum counts for reference
        # with open(os.path.join(output_dir, "train_stratum_counts.json"), 'w') as f:
        #     json.dump(train_stratum_counts, f, indent=2)
        # with open(os.path.join(output_dir, "val_stratum_counts.json"), 'w') as f:
        #     json.dump(val_stratum_counts, f, indent=2)

    except Exception as e:
        log.error(f"Failed to save subset indices: {e}")
        raise

    log.info("Stratified subset index generation complete.")
    log.info(f"Train subset size: {len(train_indices)}")
    log.info(f"Validation subset size: {len(val_indices)}")

if __name__ == "__main__":
    main() 