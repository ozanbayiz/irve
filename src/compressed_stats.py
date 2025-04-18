import h5py
import hdf5plugin # For Zstandard compression
import torch
from pathlib import Path
from sparse_autoencoder import SparseAutoencoder

USE_SAE = False # set true when wanting to use SAE model to get stats on SDFs

# -------------------------------
# Configurable constants
# -------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "datasets" / "fairface_latent_stratified.hdf5"
CACHE_FLAT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "flattened_by_race.pt"
CACHE_SAE_STATS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "compressed_reps_stats.pt"
CACHE_RAW_STATS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "encoded_reps_stats.pt"
SPLIT = "training"
NUM_RACES = 7
ENCODED_SHAPE = (577, 768)
DTYPE = torch.float16  # or float32 if memory allows


# -------------------------------
# Load encoded vectors + labels
# -------------------------------
def load_data(h5_path: Path, split="training"):
    f = h5py.File(h5_path, "r")
    encoded = f[f"{split}/encoded"]
    race_labels = f[f"{split}/labels/race"][:]
    print(f"[load_data] encoded: shape = {encoded.shape}, dtype = {encoded.dtype}")
    print(f"[load_data] race_labels: shape = {race_labels.shape}, dtype = {race_labels.dtype}")
    return encoded, race_labels


# -------------------------------
# Flatten and group by race
# -------------------------------
def flatten_and_group(encoded, race_labels, num_races=NUM_RACES):
    if CACHE_FLAT_PATH.exists():
        print("Loading cached flattened data...")
        return torch.load(CACHE_FLAT_PATH)
    print("Generating flattened race groups...")
    race_groups = [[] for _ in range(num_races)]
    for i in range(len(encoded)):
        A = torch.tensor(encoded[i], dtype=DTYPE)
        flat = A.flatten()
        race = int(race_labels[i])
        race_groups[race].append(flat)
        if i % 100 == 0:
            print(f"{i}/{len(encoded)}")
    # Stack and save
    race_groups = [torch.stack(group) for group in race_groups]
    CACHE_FLAT_PATH.parent.mkdir(exist_ok=True)
    torch.save(race_groups, CACHE_FLAT_PATH)

    return race_groups


def main():

    print(f"Loading data from: {H5_FILE}")
    encoded, race_labels = load_data(H5_FILE, SPLIT)

    print("Flattening and grouping...")
    stacked_reps = flatten_and_group(encoded, race_labels)

    for i, batch in enumerate(stacked_reps):
        print(f"Race {i}: {batch.shape[0]} samples, vector dim {batch.shape[1]}")

    if USE_SAE:    
        print("Getting SAE encodings...")
        model = SparseAutoencoder()
        model.load_state_dict(torch.load("../models/sae_weights.pt")) # real file TBD
        stacked_reps = [model.encode(b) for b in stacked_reps]
   
    # calculate stats
    means = [c.mean(dim=0) for c in stacked_reps]
    variances = [c.var(dim=0) for c in stacked_reps]

    # print out stats for each racial group
    for i in range(len(means)):
        print(f"Race {i}| mean: {means[i]}; var: {variances[i]}") 

    if USE_SAE: 
        torch.save({
            "means": means,
            "variances": variances
        }, CACHE_SAE_STATS_PATH)
    else:
        torch.save({
            "means": means,
            "variances": variances
        }, CACHE_RAW_STATS_PATH)


if __name__ == "__main__":
    main()

