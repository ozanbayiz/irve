import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import wandb
import logging
import os
import time
from .base_trainer import BaseTrainer # Assuming BaseTrainer exists
# Import dataset_worker_init_fn if defined in datasets module
from src.datasets.datasets import dataset_worker_init_fn

log = logging.getLogger(__name__)

class ProbeTrainer(BaseTrainer):
    def __init__(self,
                  cfg: DictConfig,
                  # Standard HP Names
                  learning_rate: float, # Updated name
                  batch_size: int,
                  num_epochs: int,
                  # Optional Standard HPs
                  num_workers: int = 0,
                  pin_memory: bool = True
                 ):
        super().__init__(cfg)
        self.cfg = cfg # Store full config

        # Store specific training params directly
        self.learning_rate = learning_rate # Updated name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.pin_memory = pin_memory and (self.device != torch.device('cpu')) # Adjust pin_memory

        # Initialize other attributes
        self.model = None
        self.optimizers = {}
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.tasks = self._determine_tasks() # Determine tasks based on cfg


    def _determine_tasks(self) -> list[str]:
        """Determines the list of tasks based on cfg.data.return_labels."""
        # Access return_labels from the stored full config
        requested_labels = self.cfg.data.get("return_labels")
        if requested_labels is None or not requested_labels:
            # If empty list [] was intended to mean "no labels", handle differently
            # For probes, we expect labels.
            raise ValueError("ProbeTrainer requires target tasks specified in `cfg.data.return_labels` (e.g., ['age', 'gender'] or ['all'])")
        # Special case ['all'] will be resolved during setup after dataset is loaded
        return requested_labels


    def _setup(self):
        """Instantiate model, optimizer, criterion, datasets, dataloaders."""
        log.info("Setting up Probe Trainer components...")

        # --- Data ---
        log.info(f"Instantiating dataset {self.cfg.data._target_} with return_labels={self.cfg.data.return_labels}")
        try:
            # Instantiate datasets using the full config node for data
            self.train_dataset = hydra.utils.instantiate(self.cfg.data, mode='training')
            self.val_dataset = hydra.utils.instantiate(self.cfg.data, mode='validation')
            log.info(f"Datasets created: Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
        except Exception as e:
            log.exception(f"Failed to instantiate dataset using {self.cfg.data._target_}")
            raise


        # Resolve tasks if ['all'] was requested
        if self.tasks == ['all']:
            if hasattr(self.train_dataset, 'label_keys_to_load') and self.train_dataset.label_keys_to_load:
                self.tasks = self.train_dataset.label_keys_to_load
                log.info(f"Resolved 'all' tasks to: {self.tasks}")
                if self.wandb_run:
                    wandb.config.update({"resolved_tasks": self.tasks}, allow_val_change=True)
            else:
                raise ValueError("Requested ['all'] labels, but dataset couldn't provide the available keys.")


        # Determine input dimension from the first training sample
        try:
            first_item = self.train_dataset[0]
            if isinstance(first_item, (list, tuple)):
                sample_features = first_item[0]
            else: # Should not happen if tasks are requested
                raise ValueError("Dataset did not return expected (features, labels_dict) tuple.")

            input_dim = sample_features.shape[0]
            log.info(f"Determined input dimension: {input_dim}")
            if self.wandb_run:
                wandb.config.update({"input_dim": input_dim}, allow_val_change=True)
        except Exception as e:
            log.exception("Could not determine input_dim from dataset sample.")
            raise


        # --- DataLoaders ---
        # Use instance attributes
        persistent_workers = self.num_workers > 0
        worker_init = dataset_worker_init_fn if self.num_workers > 0 and hasattr(self.train_dataset, 'file_handle') else None # Pass init_fn only if workers > 0 and dataset uses file handle

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers
        )
        log.info(f"Train Dataloader: {len(self.train_loader)} batches, {self.num_workers} workers, pin_memory={self.pin_memory}")
        log.info(f"Validation Dataloader: {len(self.val_loader)} batches")


        # --- Model ---
        log.info(f"Instantiating model: {self.cfg.model._target_}")
        try:
            self.model = hydra.utils.instantiate(
                self.cfg.model,
                input_dim=input_dim,
                # Pass class numbers explicitly from model config
                age_classes=self.cfg.model.age_classes,
                gender_classes=self.cfg.model.gender_classes,
                race_classes=self.cfg.model.race_classes
                ).to(self.device)
            log.info(f"Model '{self.model.__class__.__name__}' instantiated and moved to {self.device}.")
        except Exception as e:
             log.exception(f"Failed to instantiate model {self.cfg.model._target_}")
             raise
        # log.debug(self.model)

        # --- Optimizers & Criterion ---
        self.optimizers = {}
        # Use self.learning_rate instance attribute
        lr = self.learning_rate
        for task in self.tasks:
             classifier_attr_name = f"{task}_classifier"
             if hasattr(self.model, classifier_attr_name):
                 classifier_module = getattr(self.model, classifier_attr_name)
                 self.optimizers[task] = optim.Adam(classifier_module.parameters(), lr=lr) # Use standard lr name here
                 log.info(f"Created Adam optimizer for task '{task}' (lr={lr})")
             else:
                 log.warning(f"Model {self.model.__class__.__name__} does not have attribute '{classifier_attr_name}' for task '{task}'. Optimizer not created.")

        if not self.optimizers:
            raise ValueError(f"No optimizers created. Check if tasks {self.tasks} have corresponding attributes in model.")

        self.criterion = nn.CrossEntropyLoss()
        log.info(f"Criterion: CrossEntropyLoss")


    def _train_epoch(self, epoch: int):
        """Runs one training epoch."""
        self.model.train()
        task_losses = {task: 0.0 for task in self.tasks}
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            if not self.tasks: break
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                 log.error(f"Train: Unexpected batch format {type(batch_data)}.")
                 continue

            features, labels_dict = batch_data
            features = features.to(self.device)
            device_labels = {task: labels_dict[task].to(self.device) for task in self.tasks if task in labels_dict}

            if len(device_labels) != len(self.tasks):
                log.warning(f"Train: Batch missing labels. Expected {self.tasks}, got {list(labels_dict.keys())}. Skipping.")
                continue

            try:
                model_outputs = self.model(features)
                if not isinstance(model_outputs, tuple) or len(model_outputs) != len(self.tasks):
                    raise TypeError(f"Model output mismatch. Expected Tuple length {len(self.tasks)}, got {type(model_outputs)} len {len(model_outputs)}")
            except Exception as e:
                log.exception(f"Train: Error during model forward pass: {e}")
                continue

            num_active_tasks = len(self.tasks)
            for i, task in enumerate(self.tasks):
                if task not in self.optimizers: continue

                optimizer = self.optimizers[task]
                try:
                    preds = model_outputs[i]
                    labels = device_labels[task]
                except (IndexError, KeyError) as e:
                    log.error(f"Train: Error accessing preds/labels for task '{task}': {e}")
                    continue

                optimizer.zero_grad()
                loss = self.criterion(preds, labels)
                retain = (i < num_active_tasks - 1)
                try:
                    loss.backward(retain_graph=retain)
                    optimizer.step()
                    if task in task_losses: # Ensure task exists if loop continues after error
                        task_losses[task] += loss.item()
                except RuntimeError as e:
                    log.exception(f"Train: Runtime error backward/step task '{task}': {e}")


        log_data = {"epoch": epoch + 1}
        log_str = f"Epoch {epoch+1} Train:"
        for task in self.tasks:
            if task in task_losses and num_batches > 0:
                avg_loss = task_losses[task] / num_batches
                log_data[f"train/{task}_loss"] = avg_loss
                log_str += f" {task}_loss={avg_loss:.4f}"
        log.info(log_str)
        if self.wandb_run:
            wandb.log(log_data, step=epoch+1)


    def _validate_epoch(self, epoch: int):
        """Runs one validation epoch."""
        self.model.eval()
        task_losses = {task: 0.0 for task in self.tasks}
        task_correct = {task: 0 for task in self.tasks}
        total_samples = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_data in self.val_loader:
                if not self.tasks: break
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                    log.error(f"Val: Unexpected batch format {type(batch_data)}.")
                    continue

                features, labels_dict = batch_data
                features = features.to(self.device)
                device_labels = {task: labels_dict[task].to(self.device) for task in self.tasks if task in labels_dict}

                if len(device_labels) != len(self.tasks):
                    log.warning(f"Val: Batch missing labels. Expected {self.tasks}, got {list(labels_dict.keys())}.")
                    continue

                batch_size = features.size(0)
                total_samples += batch_size

                try:
                    model_outputs = self.model(features)
                    if not isinstance(model_outputs, tuple) or len(model_outputs) != len(self.tasks):
                         raise TypeError(f"Val: Model output mismatch. Expected Tuple length {len(self.tasks)}, got {type(model_outputs)} len {len(model_outputs)}")

                except Exception as e:
                    log.exception(f"Val: Error during model forward pass: {e}")
                    continue

                for i, task in enumerate(self.tasks):
                    try:
                        preds = model_outputs[i]
                        labels = device_labels[task]
                    except (IndexError, KeyError) as e:
                         log.error(f"Val: Error accessing preds/labels for task '{task}': {e}")
                         continue

                    loss = self.criterion(preds, labels)
                    if task in task_losses: # Check task exists
                        task_losses[task] += loss.item() * batch_size # Accumulate total loss

                    # Calculate accuracy
                    _, predicted_labels = torch.max(preds, 1)
                    if task in task_correct: # Check task exists
                        task_correct[task] += (predicted_labels == labels).sum().item()

        log_data = {"epoch": epoch + 1}
        log_str = f"Epoch {epoch+1} Val:"
        if total_samples > 0:
            for task in self.tasks:
                 avg_loss = task_losses.get(task, 0) / total_samples
                 accuracy = (task_correct.get(task, 0) / total_samples * 100)
                 log_data[f"val/{task}_loss"] = avg_loss
                 log_data[f"val/{task}_accuracy"] = accuracy
                 log_str += f" {task}_loss={avg_loss:.4f} {task}_acc={accuracy:.2f}%"
        else:
             log_str += " No samples processed."

        log.info(log_str)
        if self.wandb_run:
             wandb.log(log_data, step=epoch+1)


    def run(self):
        """Main execution method for the ProbeTrainer."""
        try:
            self._setup_wandb()
            self._setup()
        except Exception as e:
             log.exception("Setup failed. Aborting run.")
             return

        # Use self.num_epochs instance attribute
        log.info(f"Starting probe training for tasks: {self.tasks}, Epochs: {self.num_epochs}...")
        start_time = time.time()

        for epoch in range(self.num_epochs): # Use instance attribute
            epoch_start_time = time.time()
            log.info(f"--- Starting Epoch {epoch+1}/{self.num_epochs} ---")
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            log.info(f"--- Epoch {epoch+1} finished in {epoch_duration:.2f}s ---")

        total_duration = time.time() - start_time
        log.info(f"Training finished in {total_duration:.2f}s.")

        try:
            # Determine input dim for saving checkpoint
            input_dim_saved = -1
            if hasattr(self.model, 'age_classifier'):
                input_dim_saved = self.model.age_classifier.in_features
            elif hasattr(self.model, 'gender_classifier'):
                 input_dim_saved = self.model.gender_classifier.in_features
            elif hasattr(self.model, 'race_classifier'):
                 input_dim_saved = self.model.race_classifier.in_features

            # Use self.num_epochs instance attribute
            self._save_checkpoint(
                filename="probes_final.pth",
                is_final=True,
                epoch=self.num_epochs,
                input_dim=input_dim_saved,
                tasks=self.tasks
            )
        except Exception as e:
             log.exception("Failed to save final checkpoint.")
