import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import logging

log = logging.getLogger(__name__)

class PatchDatasetHDF5(Dataset):
    """
    PyTorch Dataset for loading patch embeddings from an HDF5 file.
    Assumes patches are stored contiguously for the 'train' split.
    """
    def __init__(self, hdf5_path: str, split: str = 'train'):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            split (str): Which split group to load from (e.g., 'train').
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.file = None  # Will be opened in __getitem__ if not already open
        self._data = None # HDF5 dataset object
        self.length = 0

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if self.split not in f:
                    raise ValueError(f"Split '{self.split}' not found in HDF5 file '{self.hdf5_path}'")
                if 'encoded_patches' not in f[self.split]:
                    raise ValueError(f"'encoded_patches' dataset not found in group '{self.split}'")

                self.length = f[self.split]['encoded_patches'].shape[0]
                log.info(f"Initialized PatchDatasetHDF5 for split '{self.split}' with {self.length} patches.")
        except Exception as e:
            log.error(f"Error initializing PatchDatasetHDF5: {e}")
            raise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open HDF5 file if not already open (important for multi-worker DataLoader)
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
            self._data = self.file[self.split]['encoded_patches']

        try:
            # Load the patch embedding (already float16 from saving)
            patch_embedding = self._data[idx]
            # Convert to float32 for model input
            return torch.from_numpy(patch_embedding.astype(np.float32))
        except Exception as e:
            log.error(f"Error reading index {idx} from {self.hdf5_path}: {e}")
            # Return a dummy tensor or handle appropriately
            return torch.zeros(self._data.shape[1], dtype=torch.float32) # Adjust size if needed

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is deleted
        if self.file:
            self.file.close()

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy HDF5 file for testing
    dummy_hdf5_path = 'dummy_patches.hdf5'
    n_patches = 100
    dim = 768
    with h5py.File(dummy_hdf5_path, 'w') as f:
        train_group = f.create_group('train')
        dset = train_group.create_dataset('encoded_patches', (n_patches, dim), dtype=np.float16)
        dset[:] = np.random.rand(n_patches, dim).astype(np.float16)

    print(f"Created dummy HDF5: {dummy_hdf5_path}")

    # Test the dataset
    try:
        dataset = PatchDatasetHDF5(dummy_hdf5_path, split='train')
        print(f"Dataset length: {len(dataset)}")

        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i+1}, Shape: {batch.shape}, Dtype: {batch.dtype}")
            if i >= 2: # Print a few batches
                 break

    except Exception as e:
        print(f"Error testing dataset: {e}")
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(dummy_hdf5_path):
            os.remove(dummy_hdf5_path)
            print(f"Removed dummy HDF5: {dummy_hdf5_path}") 