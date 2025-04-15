import h5py
import torch
from pathlib import Path
from sparse_autoencoder import SparseAutoencoder

# -------------------------------
# Configurable constants
# -------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data" / "mock_fairface_latent_dataset.h5"
SPLIT = "training"
NUM_RACES = 7
ENCODED_SHAPE = (576, 1024)
DTYPE = torch.float16  # or float32 if memory allows


# -------------------------------
# Load encoded vectors + labels
# -------------------------------
def load_data(h5_path: Path, split="training"):
    f = h5py.File(h5_path, "r")
    encoded = f[f"{split}/encoded"]
    race_labels = f[f"{split}/labels/race"][:]
    return encoded, race_labels


# -------------------------------
# Flatten and group by race
# -------------------------------
def flatten_and_group(encoded, race_labels, num_races=NUM_RACES):
    race_groups = [[] for _ in range(num_races)]
    for i in range(len(encoded)):
        A = torch.tensor(encoded[i], dtype=DTYPE)  # shape [576, 1024]
        flat = A.flatten()                         # shape [589824]
        race = int(race_labels[i])
        race_groups[race].append(flat)
    return race_groups


# -------------------------------
# Batch into [7, 589824] tensors
# -------------------------------
def batch_groups(race_groups):
    return [torch.stack(group) for group in race_groups]



def main():
    # load in SAE
    model = SparseAutoencoder()
    model.load_state_dict(torch.load("../models/sae_weights.pt"))

    print(f"Loading data from: {H5_FILE}")
    encoded, race_labels = load_data(H5_FILE, SPLIT)

    print("Flattening and grouping...")
    race_groups = flatten_and_group(encoded, race_labels)

    print("Batching groups...")
    batched = batch_groups(race_groups)

    for i, batch in enumerate(batched):
        print(f"Race {i}: {batch.shape[0]} samples, vector dim {batch.shape[1]}")

    compressed_reps = [model.encode(b) for b in batched]
   
    # calculate stats
    means = [c.mean(dim=0) for c in compressed_reps]
    variances = [c.var(dim=0) for c in compressed_reps]

    for i in range(len(means)):
        print(f"Race {i}| mean: {means[i]}; var: {variances[i]}") 


if __name__ == "__main__":
    main()

