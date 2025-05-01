import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os
import sys
# Ensure src is in path if running as a script/module
# (Often handled by environment or how you run it, but good practice)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)


# Setup logging
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point, configured by Hydra.

    Looks for a `script_target` key nested under the `task` key in the
    resolved configuration (expected to be set by the chosen task config)
    and executes it.
    """
    # Print the resolved configuration
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    print(f"Hydra working directory: {os.getcwd()}")
    print(f"Original working directory: {hydra.utils.get_original_cwd()}")
    print("---------------------")

    # Basic logging setup (could be more sophisticated)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # --- Get Target Function/Script ---
    # Access script_target nested under the 'task' key
    # Use .get() for safer access in case structure is unexpected
    task_config = cfg.get("task", None)
    target_path = task_config.get("script_target", None) if task_config else None
    # Or more concisely: target_path = cfg.get("task", {}).get("script_target", None)

    if target_path is None:
        log.error("Configuration Error: No 'task.script_target' key found in the resolved configuration.")
        log.error("Ensure your selected task config (e.g., config/task/your_task.yaml) defines 'script_target'.")
        sys.exit(1) # Exit cleanly

    log.info(f"Executing target: {target_path}")

    try:
        # Execute the target function/method specified in the configuration
        # hydra.utils.call dynamically imports and calls the function/method
        # passing the full configuration `cfg` as an argument.
        # IMPORTANT: We pass the *full* cfg, not just task_config,
        # so the target script has access to model, path, etc.
        hydra.utils.call(target_path, cfg=cfg) # Pass the whole cfg object

        log.info(f"Target execution ({target_path}) finished successfully.")

    except Exception as e:
        log.exception(f"An error occurred during execution of target '{target_path}':")
        # Optionally finish wandb run with failure status if it was initialized
        # Check if wandb was initialized before trying to finish
        if wandb.run is not None and wandb.run.id:
             try:
                 if wandb.run: # Check again, might have finished due to error in target
                     log.error("Finishing WandB run with failure status due to error.")
                     wandb.finish(exit_code=1)
             except Exception as wandb_e:
                 log.error(f"Error finishing WandB run: {wandb_e}")
        # Re-raise the exception after logging
        raise e

    finally:
        # --- WandB Finalization ---
        # Ensure WandB run finishes cleanly if the script completed without error
        # or if finish wasn't called in the exception block
        if wandb.run is not None and wandb.run.id:
            log.info("Finishing WandB run...")
            try:
                 if wandb.run: # Check again
                     wandb.finish()
                     log.info("WandB run finished.")
            except Exception as wandb_e:
                 log.error(f"Error finishing WandB run during final cleanup: {wandb_e}")
        # Potentially other cleanup code here


if __name__ == "__main__":
    # Make sure the 'src' directory (or project root) is importable
    # This allows hydra.utils.call to find targets like 'src.datasets...'
    # You might need to set PYTHONPATH=$PYTHONPATH:$(pwd) before running
    # or ensure your project structure allows Python to find 'src'
    main() 