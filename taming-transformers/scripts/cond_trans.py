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
train_data = DTUDataset(ROOT_DIR, "train")
val_data = DTUDataset(ROOT_DIR, "val")
model = Net2NetTransformer(**config.model.params)

# %%
import torch
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

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
   monitor='val_loss',
   patience=15,
   verbose=False,
   mode='min'
)
checkpoint_callback = ModelCheckpoint(
    dirpath='/root/autodl-tmp/checkpoints/',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
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
    checkpoint_callback=True,
    progress_bar_refresh_rate=1,
    val_check_interval = 1.0,
  

    limit_val_batches=0.1,
    log_every_n_steps=10,
    #resume_from_checkpoint="/root/autodl-tmp/checkpoints/last.ckpt"
)

# %%
lr = 1e-4
model.learning_rate = lr
trainer.fit(model, train_loader, val_loader)

# %% [markdown]
# Load the checkpoint.

# %%
import torch
ckpt_path = "logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)

# %%
model.cuda().eval()
torch.set_grad_enabled(False)

# %% [markdown]
# ## Load example data
# 
# Load an example segmentation and visualize.

# %%
from PIL import Image
import numpy as np
segmentation_path = "data/sflckr_segmentations/norway/25735082181_999927fe5a_b.png"
segmentation = Image.open(segmentation_path)
segmentation = np.array(segmentation)
segmentation = np.eye(182)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)

# %% [markdown]
# Visualize

# %%
def show_segmentation(s):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
  colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
  colorize = colorize / colorize.sum(axis=2, keepdims=True)
  s = s@colorize
  s = s[...,0,:]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  display(s)

show_segmentation(segmentation)

# %% [markdown]
# Our model also employs a VQGAN for the conditioning information, i.e. the segmentation in this example. Let's autoencode the segmentation map. Encoding returns both the quantized code and its representation in terms of indices of a learned codebook.

# %%
c_code, c_indices = model.encode_to_c(segmentation)
print("c_code", c_code.shape, c_code.dtype)
print("c_indices", c_indices.shape, c_indices.dtype)
assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[0]
segmentation_rec = model.cond_stage_model.decode(c_code)
show_segmentation(torch.softmax(segmentation_rec, dim=1))

# %% [markdown]
# Let's sample indices corresponding to codes from the image VQGAN given the segmentation code. We init randomly and take a look.

# %%
def show_image(s):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  display(s)

codebook_size = config.model.params.first_stage_config.params.embed_dim
z_indices_shape = c_indices.shape
z_code_shape = c_code.shape
z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
x_sample = model.decode_to_img(z_indices, z_code_shape)
show_image(x_sample)

# %% [markdown]
# ## Sample an image
# 
# We use the transformer in a sliding window manner to sample all code entries sequentially. The code below assumes a window size of $16\times 16$.

# %%
from IPython.display import clear_output
import time

idx = z_indices
idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])

cidx = c_indices
cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

temperature = 1.0
top_k = 100
update_every = 50

start_t = time.time()
for i in range(0, z_code_shape[2]-0):
  if i <= 8:
    local_i = i
  elif z_code_shape[2]-i < 8:
    local_i = 16-(z_code_shape[2]-i)
  else:
    local_i = 8
  for j in range(0,z_code_shape[3]-0):
    if j <= 8:
      local_j = j
    elif z_code_shape[3]-j < 8:
      local_j = 16-(z_code_shape[3]-j)
    else:
      local_j = 8

    i_start = i-local_i
    i_end = i_start+16
    j_start = j-local_j
    j_end = j_start+16
    
    patch = idx[:,i_start:i_end,j_start:j_end]
    patch = patch.reshape(patch.shape[0],-1)
    cpatch = cidx[:, i_start:i_end, j_start:j_end]
    cpatch = cpatch.reshape(cpatch.shape[0], -1)
    patch = torch.cat((cpatch, patch), dim=1)
    logits,_ = model.transformer(patch[:,:-1])
    logits = logits[:, -256:, :]
    logits = logits.reshape(z_code_shape[0],16,16,-1)
    logits = logits[:,local_i,local_j,:]

    logits = logits/temperature

    if top_k is not None:
      logits = model.top_k_logits(logits, top_k)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    idx[:,i,j] = torch.multinomial(probs, num_samples=1)

    step = i*z_code_shape[3]+j
    if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
      x_sample = model.decode_to_img(idx, z_code_shape)
      clear_output()
      print(f"Time: {time.time() - start_t} seconds")
      print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
      show_image(x_sample)

# %%



