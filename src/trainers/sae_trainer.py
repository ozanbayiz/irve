import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra # For instantiation
import wandb
import logging
import os
import time
from .base_trainer import BaseTrainer

# Import dataset_worker_init_fn if needed by your datasets
from src.datasets.datasets import dataset_worker_init_fn

log = logging.getLogger(__name__)

class SAETrainer(BaseTrainer): # Inherit from BaseTrainer
    def __init__(self,
                 cfg: DictConfig,
                 # Standard HP Names
                 learning_rate: float,
                 batch_size: int,
                 num_epochs: int,
                 # SAE Specific HP
                 l1: float,
                 # Optional Standard HPs
                 num_workers: int = 0,
                 pin_memory: bool = True
                ):
        super().__init__(cfg) # Pass full cfg to BaseTrainer
        self.cfg = cfg # Still store full config if needed

        # Store specific training params directly
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.l1 = l1
        self.num_workers = num_workers
        self.pin_memory = pin_memory and (self.device != torch.device('cpu')) # Adjust pin_memory based on device

        # Initialize other attributes (model, optimizer, etc.) to None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None

    def _setup(self):
        """Instantiate model, optimizer, criterion, datasets, dataloaders."""
        log.info("Setting up SAE Trainer components...")

        # --- Data ---
        log.info(f"Instantiating dataset using config: {self.cfg.data._target_}")
        try:
            # Assume the configured dataset class accepts a 'mode' argument
            # No special handling for MNIST is needed anymore.
            self.train_dataset = hydra.utils.instantiate(self.cfg.data, mode='training')
            self.val_dataset = hydra.utils.instantiate(self.cfg.data, mode='validation')
            log.info(f"Datasets created: Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
        except Exception as e:
             log.exception(f"Failed to instantiate dataset using {self.cfg.data._target_}. Ensure it accepts a 'mode' argument ('training'/'validation').")
             raise

        # --- DataLoaders ---
        # Use instance attributes for configuration
        persistent_workers = self.num_workers > 0
        # Pass worker_init_fn only if workers > 0 and the dataset likely needs it (has file_handle)
        worker_init = dataset_worker_init_fn if self.num_workers > 0 and hasattr(self.train_dataset, 'file_handle') else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init
        )
        log.info(f"Train Dataloader: {len(self.train_loader)} batches, {self.num_workers} workers, pin_memory={self.pin_memory}")
        log.info(f"Validation Dataloader: {len(self.val_loader)} batches")

        # --- Model ---
        log.info(f"Instantiating model: {self.cfg.model._target_}")
        # Determine input_size dynamically from dataset if not explicitly set in model config
        if not hasattr(self.cfg.model, 'input_size') or self.cfg.model.input_size is None:
             try:
                first_item = self.train_dataset[0]
                if isinstance(first_item, (list, tuple)):
                    sample_features = first_item[0].flatten()
                else:
                    sample_features = first_item.flatten()
                input_size = sample_features.shape[0] # Assumes flattened features
                log.info(f"Determined input_size from dataset: {input_size}")
                # Update the model config node or pass directly to instantiation
                # Passing directly is cleaner here:
                self.model = hydra.utils.instantiate(self.cfg.model, input_size=input_size).to(self.device)
             except Exception as e:
                 log.exception("Could not determine input_size from dataset. Please set model.input_size in the config.")
                 raise
        else:
             # Use input_size from config
             log.info(f"Using input_size from config: {self.cfg.model.input_size}")
             self.model = hydra.utils.instantiate(self.cfg.model).to(self.device)

        log.info(f"Model '{self.model.__class__.__name__}' instantiated and moved to {self.device}.")
        # log.debug(self.model)

        # --- Optimizer & Criterion ---
        # Use self.learning_rate instance attribute
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        log.info(f"Optimizer: Adam (lr={self.learning_rate})")
        log.info(f"Criterion: MSELoss")

    def _train_epoch(self, epoch: int):
        """Runs one training epoch."""
        self.model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle dataset returning (features,) or (features, labels_dict)
            if isinstance(batch_data, (list, tuple)):
                 batch = batch_data[0] # Assume features are the first element
            else:
                 batch = batch_data # Assume only features are returned

            # Ensure data is flattened (SAE operates on vectors)
            batch = batch.view(batch.size(0), -1).to(self.device)

            self.optimizer.zero_grad()
            recon, encoded = self.model(batch.reshape(batch.size(0), -1))

            recon_loss = self.criterion(recon, batch)
            # Use self.l1 instance attribute
            l1_loss = self.l1 * torch.norm(encoded, 1)
            loss = recon_loss + l1_loss

            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
        avg_l1_loss = total_l1_loss / num_batches if num_batches > 0 else 0

        log.info(f"Epoch {epoch+1} Train: Avg Loss={avg_train_loss:.4f}, Avg Recon Loss={avg_recon_loss:.4f}, Avg L1 Loss={avg_l1_loss:.4f}")
        if self.wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/recon_loss": avg_recon_loss,
                "train/l1_loss": avg_l1_loss
            }, step=epoch+1)

    def _validate_epoch(self, epoch: int):
        """Runs one validation epoch."""
        self.model.eval()
        total_val_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_data in self.val_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch = batch_data[0]
                else:
                    batch = batch_data

                batch = batch.view(batch.size(0), -1).to(self.device)
                recon, encoded = self.model(batch)

                recon_loss = self.criterion(recon, batch)
                # Use self.l1 instance attribute
                l1_loss = self.l1 * torch.norm(encoded, 1)
                loss = recon_loss + l1_loss

                total_val_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_l1_loss += l1_loss.item()

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
        avg_l1_loss = total_l1_loss / num_batches if num_batches > 0 else 0

        log.info(f"Epoch {epoch+1} Val: Avg Loss={avg_val_loss:.4f}, Avg Recon Loss={avg_recon_loss:.4f}, Avg L1 Loss={avg_l1_loss:.4f}")
        if self.wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/recon_loss": avg_recon_loss,
                "val/l1_loss": avg_l1_loss
            }, step=epoch+1)

    def run(self):
        """Main execution method for the SAETrainer."""
        try:
            self._setup_wandb()
            self._setup()
        except Exception as e:
             log.exception("Setup failed. Aborting run.")
             # Optionally finish wandb run with error state
             # if self.wandb_run: wandb.finish(exit_code=1)
             return # Exit if setup fails

        # Use self.num_epochs instance attribute
        log.info(f"Starting SAE training for {self.num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(self.num_epochs): # Use instance attribute
            epoch_start_time = time.time()
            log.info(f"--- Starting Epoch {epoch+1}/{self.num_epochs} ---")
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            log.info(f"--- Epoch {epoch+1} finished in {epoch_duration:.2f}s ---")
            # Add checkpointing logic here if desired (e.g., save best model based on val_loss)

        total_duration = time.time() - start_time
        log.info(f"Training finished in {total_duration:.2f}s.")

        # Save final model
        try:
            # Use self.num_epochs instance attribute
            self._save_checkpoint(filename="sae_final.pth", is_final=True, epoch=self.num_epochs)
        except Exception as e:
            log.exception("Failed to save final checkpoint.")

        # Finish wandb run (if initialized by this trainer/base class)
        # if self.wandb_run:
        #    log.info("Finishing WandB run.")
        #    wandb.finish()