# %% [markdown]
# # Taming Transformers
# 
# This notebook is a minimal working example to generate landscape images as in [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers). **tl;dr** We combine the efficiancy of convolutional approaches with the expressivity of transformers by introducing a convolutional VQGAN, which learns a codebook of context-rich visual parts, whose composition is modeled with an autoregressive transformer.

# %% [markdown]
# ## Setup
# The setup code in this section was written to be [run in a Colab environment](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/taming-transformers.ipynb). For a full, local setup, we recommend the provided [conda environment](https://github.com/CompVis/taming-transformers/blob/master/environment.yaml), as [described in the readme](https://github.com/CompVis/taming-transformers#requirements). This will also allow you to run a streamlit based demo.
# 
# Here, we first clone the repository and download a model checkpoint and config.

# %%

# !mkdir -p logs/2020-11-09T13-31-51_sflckr/checkpoints
# !wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt'
# !mkdir logs/2020-11-09T13-31-51_sflckr/configs
# !wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml'

# %% [markdown]
# Next, we install minimal required dependencies.

# %%
# %pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 einops transformers
# import sys
# sys.path.append(".")

# %% [markdown]
# ## Loading the model
# 
# We load and print the config.

# %%
from omegaconf import OmegaConf
config_path = "/root/autodl-tmp/taming-transformers/scripts/logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
config = OmegaConf.load(config_path)
import yaml
print(yaml.dump(OmegaConf.to_container(config)))

# %% [markdown]
# Instantiate the model.

# %%
import sys
sys.path.append("/root/autodl-tmp/taming-transformers/")
sys.path.append("/root/autodl-tmp/project/dp_simple/CasMVSNet_pl")
from datasets.dtu import DTUDataset
from taming.models.cond_transformer import Net2NetTransformer
ROOT_DIR = "/root/autodl-tmp/mvs_training/dtu/"
train_data = DTUDataset(ROOT_DIR, "train",img_wh=(512,640))
val_data = DTUDataset(ROOT_DIR, "val",img_wh=(512,640))
model = Net2NetTransformer(**config.model.params)

# %%
import torch
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

# %%
import torch
imgs = torch.randn(2, 3, 512, 640)

# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

logger = TensorBoardLogger("/root/autodl-tmp/log/", name="transformer")
early_stop_callback = EarlyStopping(
   monitor='val/loss_diff',
   patience=15,
   verbose=False,
   mode='min'
)
checkpoint_callback = ModelCheckpoint(
    dirpath='/root/autodl-tmp/checkpoints/',
    
    save_top_k=1,
    verbose=True,
    monitor='val/loss_diff',
    mode='min',
    save_last=True,
    # define the filename
    filename='sample-{epoch:02d}-{val_loss:.2f}'
    
)
trainer = Trainer(
    logger=logger,
    max_epochs=100,
    gpus=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    
    progress_bar_refresh_rate=1,
    val_check_interval = 1.0,
    
    precision=16,
    
   
    log_every_n_steps=10,
    #resume_from_checkpoint="/root/autodl-tmp/checkpoints/last.ckpt"
)

# %%
for batch in train_loader:
    print(batch['imgs'].shape)
    break

# %%
lr = 1e-4
model.learning_rate = lr
trainer.fit(model, train_loader, val_loader)
