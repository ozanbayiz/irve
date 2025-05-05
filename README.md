# Investigating Demographic Attribute Representation in Vision Encoders (IDARVE)

### Setting up the Virtual Environment
```
conda env create -f environment.yaml
```

### Loading the Checkpoints
```
bash setup/download_checkpoints.sh
```

### Setting up Datasets
from the repo's directory, run
```
python -m src.setup_datasets.ve_latent_dataset
```
```
python -m src.setup_datasets.sae_latent_dataset
```