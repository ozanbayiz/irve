import torch
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import logging

log = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._setup_device()
        self.wandb_run = None # To store the initialized wandb run

    def _setup_device(self) -> torch.device:
        """Sets up the device (CPU or CUDA) based on config and availability."""
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        log.info(f"Using device: {device}")
        return device

    def _setup_wandb(self) -> None:
        """Initializes Weights & Biases run."""
        if self.wandb_run is None: # Initialize only once
            log.info(f"Initializing WandB run... Project: {self.cfg.wandb.project}")
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.get("entity"), # Use .get for optional keys
                config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                # name=hydra.core.hydra_config.HydraConfig.get().job.name # Optional: Use Hydra job name
                # dir=os.getcwd() # Ensure logs are saved in hydra output dir
            )
            log.info(f"WandB Run Name: {self.wandb_run.name}, ID: {self.wandb_run.id}")
        else:
            log.warning("WandB already initialized.")


    def _save_checkpoint(self, filename: str = "checkpoint.pth", **kwargs) -> None:
        """Saves model and potentially optimizer state."""
        # Ensure model is on CPU before saving to avoid GPU mapping issues on load
        self.model.cpu()
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            # Add optimizers dict for probes if needed
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            **kwargs # Allow saving extra info like epoch
        }
        save_path = os.path.join(os.getcwd(), filename) # Save in Hydra output dir
        torch.save(state, save_path)
        log.info(f"Checkpoint saved to {save_path}")
        # Move model back to original device
        self.model.to(self.device)

        # Log as artifact if enabled
        if self.cfg.wandb.log_model and self.wandb_run:
             artifact = wandb.Artifact(name=f"{self.wandb_run.id}-{filename.split('.')[0]}", type="model")
             artifact.add_file(save_path)
             self.wandb_run.log_artifact(artifact)
             log.info(f"Checkpoint logged as WandB artifact: {artifact.name}")


    # Placeholder for the main run method, to be implemented by subclasses
    def run(self):
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    # Optional: _load_checkpoint method if needed
