import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR # For linear warmup + cosine decay
import wandb
import os
import time
import math
import logging
from tqdm.auto import tqdm

# Assuming models and datasets are importable from src
from src.models.sparse_autoencoder import SparseAutoencoder
from src.latent_datasets.patch_dataset import PatchDatasetHDF5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * num_cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@hydra.main(config_path="../../config", config_name="task/train_sae", version_base=None)
def main(cfg: DictConfig):
    log.info("Starting SAE Patch Training...")
    log.info(f"Full Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- Setup ---
    # Device
    if cfg.trainer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.trainer.device)
    log.info(f"Using device: {device}")

    # Wandb
    run_name = f"{cfg.wandb.run_name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity, # Ensure this is set in config!
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), # Log resolved config
            name=run_name,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
            # dir=cfg.wandb.dir # Optional: specify wandb log dir
        )
        log.info(f"Wandb initialized for run: {run_name} (ID: {wandb.run.id})")
    except Exception as e:
        log.error(f"Failed to initialize Wandb: {e}. Check entity/project settings.")
        log.warning("Proceeding without Wandb logging.")
        wandb.init(mode="disabled") # Ensure wandb calls don't crash

    # --- Data ---
    hdf5_path = cfg.path.patch_embeddings_file
    log.info(f"Loading training patches from: {hdf5_path}")
    try:
        train_dataset = PatchDatasetHDF5(hdf5_path, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=4, # Adjust based on system
            pin_memory=True,
            drop_last=True # Good practice for consistent batch stats
        )
        log.info(f"Created DataLoader with batch size {cfg.trainer.batch_size}, {len(train_loader)} batches per epoch.")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        wandb.finish(exit_code=1)
        return

    # --- Model ---
    log.info("Initializing SAE model...")
    try:
        model = SparseAutoencoder(cfg.model).to(device)
        log.info(f"Model:\n{model}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters: {num_params:,}")
        wandb.config.update({"model/total_params": num_params}) # Log param count
        wandb.watch(model, log='all', log_freq=cfg.trainer.log_interval * 10) # Watch model gradients/params
    except Exception as e:
        log.error(f"Failed to initialize model: {e}")
        wandb.finish(exit_code=1)
        return

    # --- Optimizer and Scheduler ---
    log.info("Setting up optimizer and scheduler...")
    try:
        # Instantiate optimizer using Hydra's instantiation
        optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, params=model.parameters())
        log.info(f"Optimizer: {optimizer}")

        # Determine total training steps
        if hasattr(cfg.trainer, "max_steps") and cfg.trainer.max_steps:
            max_steps = cfg.trainer.max_steps
            num_epochs = math.ceil(max_steps / len(train_loader))
            log.info(f"Training for max_steps: {max_steps}")
        elif hasattr(cfg.trainer, "num_epochs") and cfg.trainer.num_epochs:
            num_epochs = cfg.trainer.num_epochs
            max_steps = num_epochs * len(train_loader)
            log.info(f"Training for num_epochs: {num_epochs} ({max_steps} steps)")
        else:
            raise ValueError("Must define either trainer.max_steps or trainer.num_epochs in config.")
        wandb.config.update({"training/max_steps": max_steps, "training/num_epochs": num_epochs})

        # Setup LR scheduler if configured
        scheduler = None
        if cfg.trainer.lr_scheduler.use_scheduler:
             num_warmup_steps = cfg.trainer.lr_scheduler.warmup_steps
             scheduler = get_cosine_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=num_warmup_steps,
                 num_training_steps=max_steps
             )
             log.info(f"Using Cosine LR scheduler with {num_warmup_steps} warmup steps.")

    except Exception as e:
        log.error(f"Failed to setup optimizer/scheduler: {e}")
        wandb.finish(exit_code=1)
        return

    # --- Checkpointing Setup ---
    checkpoint_dir = cfg.trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- Training Loop ---
    log.info("Starting training loop...")
    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        log.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if global_step >= max_steps:
                log.info("Reached max_steps. Stopping training.")
                break

            optimizer.zero_grad()

            # Move batch to device
            inputs = batch.to(device)

            # Forward pass
            reconstruction, hidden_activations = model(inputs)

            # Calculate loss
            total_loss, recon_loss, sparsity_loss = model.calculate_loss(
                inputs, reconstruction, hidden_activations, cfg.trainer.lambda_sparsity
            )

            # Backward pass and optimize
            total_loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update LR scheduler (per step)
            if scheduler:
                scheduler.step()

            # --- Logging ---
            if global_step % cfg.trainer.log_interval == 0:
                l0_norm = model.calculate_l0_norm(hidden_activations)
                current_lr = optimizer.param_groups[0]['lr']

                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/reconstruction_loss": recon_loss.item(),
                    "train/sparsity_loss_L1": sparsity_loss.item(), # Log the L1 value itself
                    "metrics/L0_norm": l0_norm.item(),
                    "hyperparameters/learning_rate": current_lr,
                    "progress/epoch": epoch + (batch_idx / len(train_loader)),
                    "progress/global_step": global_step
                }, step=global_step)

                progress_bar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "L0": f"{l0_norm.item():.2f}",
                    "LR": f"{current_lr:.2e}"
                })

            # --- Checkpointing ---
            if global_step > 0 and global_step % cfg.trainer.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}.pth")
                log.info(f"Saving checkpoint to {checkpoint_path}...")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': OmegaConf.to_container(cfg, resolve=True) # Save config with checkpoint
                }, checkpoint_path)
                # Optional: Save checkpoint to wandb artifacts
                # artifact = wandb.Artifact(f'sae_checkpoint_step_{global_step}', type='model')
                # artifact.add_file(checkpoint_path)
                # wandb.log_artifact(artifact)


            global_step += 1

        if global_step >= max_steps:
            break # Exit outer loop if max_steps reached mid-epoch

    # --- Final Save ---
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    log.info(f"Training finished. Saving final model to {final_checkpoint_path}")
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True) # Save config
    }, final_checkpoint_path)
    # Optional: Save final model to wandb artifacts
    # final_artifact = wandb.Artifact('sae_final_model', type='model', metadata={'final_step': global_step})
    # final_artifact.add_file(final_checkpoint_path)
    # wandb.log_artifact(final_artifact, aliases=['latest', f'step_{global_step}'])


    wandb.finish()
    log.info("SAE Patch Training complete.")


if __name__ == "__main__":
    main() 