import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import numpy as np
import h5py
import os
from tqdm.auto import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming model and dataset structures are importable
from src.models.sparse_autoencoder import SparseAutoencoder
# Note: We might not need the PatchDatasetHDF5 if loading all activations at once,
# but a simpler dataset for batched activation calculation is useful.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Helper Dataset for Activation Calculation ---
class ActivationDataset(Dataset):
    def __init__(self, hdf5_path, split_group, dataset_name="encoded_patches"):
        self.hdf5_path = hdf5_path
        self.split_group = split_group
        self.dataset_name = dataset_name
        self.file = None
        self._data = None
        self.length = 0
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.length = f[self.split_group][self.dataset_name].shape[0]
        except Exception as e:
            log.error(f"Failed to get length from HDF5: {e}")
            raise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
            self._data = self.file[self.split_group][self.dataset_name]
        patch = self._data[idx]
        return torch.from_numpy(patch.astype(np.float32))

    def __del__(self):
        if self.file:
            self.file.close()

# --- Analysis Functions ---
@torch.no_grad()
def compute_activations(model, dataloader, device):
    """Computes SAE hidden activations for all data in the dataloader."""
    all_activations = []
    model.eval()
    model.to(device)
    for batch in tqdm(dataloader, desc="Computing SAE Activations"):
        inputs = batch.to(device)
        hidden_activations = model.encode(inputs)
        all_activations.append(hidden_activations.cpu()) # Move to CPU to save GPU memory
    return torch.cat(all_activations, dim=0).numpy() # Return as numpy array

def perform_statistical_tests(activations, labels, label_name, correction_method='fdr_bh', alpha=0.05):
    """
    Performs t-tests (binary) or ANOVA (multi-class) for each feature.

    Args:
        activations (np.ndarray): Shape (n_samples, n_features).
        labels (np.ndarray): Shape (n_samples,). Integer labels.
        label_name (str): Name of the label being tested (for logging).
        correction_method (str): Method for multiple testing correction.
        alpha (float): Significance level.

    Returns:
        pd.DataFrame: DataFrame with results per feature.
    """
    n_samples, n_features = activations.shape
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    log.info(f"Performing tests for '{label_name}' with {n_groups} groups: {unique_labels}")

    results = []
    p_values = np.ones(n_features) * np.nan # Initialize with NaN

    if n_groups < 2:
        log.warning(f"Skipping tests for '{label_name}': only {n_groups} unique label found.")
        return pd.DataFrame()
    elif n_groups == 2:
        # Independent T-test (Welch's t-test for unequal variances)
        group0_indices = (labels == unique_labels[0])
        group1_indices = (labels == unique_labels[1])
        if not np.any(group0_indices) or not np.any(group1_indices):
            log.warning(f"Skipping T-tests for '{label_name}': one group has no samples.")
            return pd.DataFrame()

        activations_group0 = activations[group0_indices, :]
        activations_group1 = activations[group1_indices, :]
        # Perform t-test feature by feature
        for feature_idx in tqdm(range(n_features), desc=f"T-tests for {label_name}", leave=False):
            # Check for variance before test
            if np.var(activations_group0[:, feature_idx]) > 1e-9 or np.var(activations_group1[:, feature_idx]) > 1e-9:
                 try:
                     stat, p = stats.ttest_ind(
                         activations_group0[:, feature_idx],
                         activations_group1[:, feature_idx],
                         equal_var=False, # Welch's t-test
                         nan_policy='omit' # Handle potential NaNs if they exist
                     )
                     p_values[feature_idx] = p
                     mean_diff = np.nanmean(activations_group1[:, feature_idx]) - np.nanmean(activations_group0[:, feature_idx])
                     results.append({'feature_index': feature_idx, 'statistic': stat, 'p_value': p, 'mean_diff': mean_diff})
                 except Exception as e:
                    log.warning(f"T-test failed for feature {feature_idx} on label {label_name}: {e}")
                    results.append({'feature_index': feature_idx, 'statistic': np.nan, 'p_value': np.nan, 'mean_diff': np.nan})
            else:
                 # Handle zero variance case
                 p_values[feature_idx] = 1.0 # No difference if no variance
                 results.append({'feature_index': feature_idx, 'statistic': 0.0, 'p_value': 1.0, 'mean_diff': 0.0})

        test_type = 't-test'

    else:
        # One-way ANOVA
        groups_data = [activations[labels == lbl, :] for lbl in unique_labels]
        if any(g.shape[0] == 0 for g in groups_data):
             log.warning(f"Skipping ANOVA for '{label_name}': at least one group has no samples.")
             return pd.DataFrame()
        # Perform ANOVA feature by feature
        for feature_idx in tqdm(range(n_features), desc=f"ANOVA for {label_name}", leave=False):
             feature_groups = [group[:, feature_idx] for group in groups_data]
             # Check for variance within groups
             if all(np.var(fg) > 1e-9 for fg in feature_groups if len(fg) > 0):
                 try:
                     stat, p = stats.f_oneway(*feature_groups)
                     p_values[feature_idx] = p
                     # Calculate effect size (eta-squared) - simplified version
                     overall_mean = np.mean(activations[:, feature_idx])
                     ss_between = sum(len(fg) * (np.mean(fg) - overall_mean)**2 for fg in feature_groups)
                     ss_total = np.sum((activations[:, feature_idx] - overall_mean)**2)
                     eta_sq = ss_between / ss_total if ss_total > 1e-9 else 0
                     results.append({'feature_index': feature_idx, 'statistic': stat, 'p_value': p, 'eta_sq': eta_sq})
                 except Exception as e:
                     log.warning(f"ANOVA failed for feature {feature_idx} on label {label_name}: {e}")
                     results.append({'feature_index': feature_idx, 'statistic': np.nan, 'p_value': np.nan, 'eta_sq': np.nan})

             else:
                 # Handle zero variance cases
                 p_values[feature_idx] = 1.0
                 results.append({'feature_index': feature_idx, 'statistic': 0.0, 'p_value': 1.0, 'eta_sq': 0.0})


        test_type = 'ANOVA'

    results_df = pd.DataFrame(results)
    if results_df.empty:
        log.warning(f"No results generated for {label_name} tests.")
        return results_df

    # Multiple comparison correction
    valid_p_indices = ~np.isnan(p_values)
    if np.any(valid_p_indices) and correction_method:
        valid_pvals = p_values[valid_p_indices]
        reject, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=alpha, method=correction_method)
        corrected_p = np.ones_like(p_values) * np.nan
        corrected_p[valid_p_indices] = pvals_corrected
        significant = np.zeros_like(p_values, dtype=bool)
        significant[valid_p_indices] = reject

        results_df['p_value_corrected'] = corrected_p
        results_df['significant'] = significant
        log.info(f"Applied {correction_method} correction. Found {significant.sum()} significant features for {label_name}.")
    else:
        results_df['p_value_corrected'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < alpha if alpha else False
        log.info(f"No multiple test correction applied or no valid p-values. Found {results_df['significant'].sum()} significant features for {label_name} (raw p < {alpha}).")


    results_df['test_type'] = test_type
    results_df = results_df.sort_values('p_value_corrected', ascending=True)

    return results_df

# --- Main Analysis Function ---
@hydra.main(config_path="../../config", config_name="task/analyze_sae", version_base=None)
def main(cfg: DictConfig):
    log.info("Starting SAE Correlation Analysis...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- Setup ---
    if cfg.analysis.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.analysis.device)
    log.info(f"Using device: {device}")

    output_dir = cfg.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Saving results to: {output_dir}")

    # --- Load Model ---
    # Determine checkpoint path (priority: explicit path > wandb)
    checkpoint_path = cfg.analysis.get('sae_checkpoint_path', None)
    wandb_run_id = cfg.analysis.get('wandb_run_id', None)

    if checkpoint_path:
        log.info(f"Loading model from checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            log.error(f"Checkpoint file not found: {checkpoint_path}")
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
            # Infer model config from checkpoint if available, else use task config
            model_cfg = OmegaConf.create(checkpoint.get('config', {}).get('model', cfg.model))
            model = SparseAutoencoder(model_cfg)
            model.load_state_dict(checkpoint['model_state_dict'])
            log.info("SAE model loaded successfully from local checkpoint.")
        except Exception as e:
            log.error(f"Failed to load model from {checkpoint_path}: {e}")
            return
    elif wandb_run_id:
        # --- Wandb Artifact Loading Logic (Requires wandb API) ---
        log.warning("Wandb artifact loading not implemented in this script. Please load manually or provide local path.")
        # Example sketch:
        # import wandb
        # api = wandb.Api()
        # run = api.run(wandb_run_id) # e.g., "entity/project/run_id"
        # artifact_name = f'run-{run.id}-sae_final_model' # Default name if saved that way
        # artifact_alias = cfg.analysis.wandb_artifact_alias
        # artifact = run.use_artifact(f'{artifact_name}:{artifact_alias}')
        # artifact_dir = artifact.download()
        # checkpoint_path = os.path.join(artifact_dir, "final_model.pth")
        # # ... then load from checkpoint_path as above ...
        # --- End Sketch ---
        return # Exit because not implemented
    else:
        log.error("No valid 'sae_checkpoint_path' or 'wandb_run_id' provided in config.")
        return

    model.to(device)
    model.eval()

    # --- Load Data ---
    hdf5_path = cfg.analysis.hdf5_path
    split_group = cfg.analysis.validation_split_group
    log.info(f"Loading validation data from: {hdf5_path} [{split_group}]")
    try:
        # Load Labels into memory
        with h5py.File(hdf5_path, 'r') as f:
            labels_group = f[split_group]['labels']
            labels = {}
            for label_name in labels_group.keys():
                 # Flatten labels as they are (N*576, 1) -> (N*576,)
                 labels[label_name] = labels_group[label_name][:].flatten()
                 log.info(f"Loaded label '{label_name}' with shape {labels[label_name].shape}")
        labels_df = pd.DataFrame(labels)

        # Create dataset/dataloader for activations
        activation_dataset = ActivationDataset(hdf5_path, split_group)
        activation_loader = DataLoader(
            activation_dataset,
            batch_size=cfg.analysis.activation_batch_size,
            shuffle=False,
            num_workers=2 # Can use fewer workers for analysis
        )
        n_samples = len(activation_dataset)
        n_features = model.hidden_dim # Assuming hidden_dim is number of features
        log.info(f"Validation set size: {n_samples} patches, {n_features} SAE features.")

    except Exception as e:
        log.error(f"Failed to load validation data: {e}")
        return

    if n_samples != len(labels_df):
         log.error(f"Mismatch between number of patches ({n_samples}) and labels ({len(labels_df)}) loaded.")
         return

    # --- Compute Activations ---
    log.info("Computing SAE activations for validation set...")
    activations = compute_activations(model, activation_loader, device) # Shape: (n_samples, n_features)
    log.info(f"Computed activations shape: {activations.shape}")

    # --- Perform Analysis ---
    all_results = {}
    if cfg.analysis.method == 'stats_test':
        log.info("Performing statistical tests (t-test/ANOVA)...")
        for label_name in labels_df.columns:
             results_df = perform_statistical_tests(
                 activations,
                 labels_df[label_name].values,
                 label_name,
                 correction_method=cfg.analysis.multiple_test_correction,
                 alpha=cfg.analysis.alpha
             )
             if not results_df.empty:
                 all_results[label_name] = results_df
                 # Save results to CSV
                 output_csv_path = os.path.join(output_dir, f"{cfg.analysis.results_prefix}_{label_name}.csv")
                 results_df.to_csv(output_csv_path, index=False)
                 log.info(f"Saved results for '{label_name}' to {output_csv_path}")

    elif cfg.analysis.method == 'linear_probe':
        log.warning("Linear probe analysis method not implemented yet.")
        # TODO: Implement linear probing (train/eval sklearn models)
        pass
    else:
        log.error(f"Unsupported analysis method: {cfg.analysis.method}")

    # --- Visualize Top Features (Example: Distribution plots) ---
    log.info("Generating visualization for top features...")
    top_n = cfg.analysis.top_n_features
    sns.set_theme(style="whitegrid")

    for label_name, results_df in all_results.items():
        significant_results = results_df[results_df['significant'] == True]
        if significant_results.empty:
            log.info(f"No significant features found for {label_name} to visualize.")
            continue

        n_to_plot = min(top_n, len(significant_results))
        top_features = significant_results.head(n_to_plot)['feature_index'].tolist()
        log.info(f"Visualizing top {n_to_plot} significant features for {label_name}: {top_features}")

        # Create DataFrame for plotting
        plot_df = labels_df[[label_name]].copy()
        plot_df['label'] = plot_df[label_name] # Use original label column name
        for feat_idx in top_features:
             plot_df[f'feature_{feat_idx}'] = activations[:, feat_idx]

        # Plot distributions (e.g., violin plots)
        n_cols = min(n_to_plot, 4)
        n_rows = (n_to_plot + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
        axes = axes.flatten()

        for i, feat_idx in enumerate(top_features):
            stat_info = significant_results[significant_results['feature_index'] == feat_idx].iloc[0]
            p_corr = stat_info['p_value_corrected']
            title = f"Feature {feat_idx}\n(p_corr={p_corr:.2e})"
            if 'mean_diff' in stat_info: title += f"\nMeanDiff={stat_info['mean_diff']:.2f}"
            if 'eta_sq' in stat_info: title += f"\nEtaSq={stat_info['eta_sq']:.2f}"

            try:
                sns.violinplot(ax=axes[i], data=plot_df, x='label', y=f'feature_{feat_idx}')
                axes[i].set_title(title)
                axes[i].set_xlabel(label_name)
                axes[i].set_ylabel("Activation")
            except Exception as e:
                 log.warning(f"Plotting failed for feature {feat_idx}, label {label_name}: {e}")
                 axes[i].set_title(f"Feature {feat_idx}\nPlotting Error")


        # Hide unused axes
        for j in range(n_to_plot, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Top {n_to_plot} Significant SAE Feature Activations vs. {label_name}", y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{cfg.analysis.results_prefix}_{label_name}_top_features.png")
        plt.savefig(plot_path, bbox_inches='tight')
        log.info(f"Saved feature plot for {label_name} to {plot_path}")
        plt.close(fig)

    log.info("Analysis script finished.")

if __name__ == "__main__":
    main() 