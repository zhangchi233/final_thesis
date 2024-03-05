# %%
from typing import Any, Optional
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
# %%
import torch.nn.functional as F
import numpy as np
import tqdm
import subprocess
import os
from einops import rearrange
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
# import tensorboard 
from torch.utils.tensorboard import SummaryWriter
# import checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
# %%
import os
from diffusers import VQModel,UNet2DModel,DDIMScheduler

# %%
class TripletImageDataset(Dataset):
    def __init__(self, img_triplets, labels, transform=None,vae=None,scheduler=None):
        """
        img_triplets: List of tuples (img_path1, img_path2, img_path3)
        labels: List of binary labels
        transform: torchvision transforms for preprocessing
        """
        self.img_triplets = img_triplets
        self.labels = labels
        self.transform = transform
        self.vae = vae.cuda()
        self.scheduler = scheduler

    def __len__(self):
        return len(self.img_triplets)
    @torch.no_grad()
    def __getitem__(self, idx):
        #img1, img2, img3 = self.img_triplets[idx]
        img = self.img_triplets[idx]
        label = self.labels[idx]

        # Load images and apply transformations
        try:
            img = self.transform(Image.open(img).resize((512, 512)))*2-1
            #img1 = self.transform(Image.open(img1).resize((512, 512)))*2-1
            #img2 = self.transform(Image.open(img2).resize((512, 512)))*2-1
            #img3 = self.transform(Image.open(img3).resize((512, 512)))*2-1
        except:
            print("Error loading image")
            print(self.img_triplets[idx])
            return self.__getitem__(idx + 1)
        
        
        
        

        #imgs = torch.stack([img1, img2, img3], dim=0).float()
        
        label = torch.tensor(label)

        return img, label


# %%


def build_metas(path,examples):
    files = os.listdir(path)
    
    #img_format
    #define positive examples
    labels = []
    metas = []
    i=0
    for file in files:
        print(file," file select is")

        images = os.listdir(os.path.join(path,file))
        
        for img in images:
            if img.endswith(".png") and "rect_" in img:
                metas.append(os.path.join(path,file,img))
                label = img.split("_")[2]
                labels.append([int(label)])
        break
    
    '''
    while i < examples:
        for file in files:
            
            # randomly select 3 views from 0-49
            views = np.random.choice(list(range(1,50)), 3, replace=False)
            # randomly select one light from 0-7
            light = np.random.choice(7, 1, replace=False)[0]
            file_name = "rect_{:03d}_{}_r5000.png".format(views[0], light)
            img1 = os.path.join(path,file,file_name)
            img2 = os.path.join(path,file,"rect_{:03d}_{}_r5000.png".format(views[1], light))
            img3 = os.path.join(path,file,"rect_{:03d}_{}_r5000.png".format(views[2], light))
            #if os.path.exists(img1) and os.path.exists(img2) and os.path.exists(img3):
            metas.append((img1,img2,img3))
            labels.append([1])
        i+= 1
    #define negative examples
    i = 0
    while i < examples:
        for file in files:
            
            # randomly select 3 views from 1-49
            views = np.random.choice(list(range(1,50)), 3, replace=False)
            # randomly select one light from 0-6
            light = np.random.choice(7, 1, replace=False)[0]
            # select a different light condition
            lights = [i for i in range(7) if i != light]
            light_neg = np.random.choice(lights, 1, replace=False)[0]


            file_name = "rect_{:03d}_{}_r5000.png".format(views[0], light_neg)
            img1 = os.path.join(path,file,file_name)
            img2 = os.path.join(path,file,"rect_{:03d}_{}_r5000.png".format(views[1], light))
            img3 = os.path.join(path,file,"rect_{:03d}_{}_r5000.png".format(views[2], light))
            #if os.path.exists(img1) and os.path.exists(img2) and os.path.exists(img3):
            metas.append((img1,img2,img3))
            labels.append([0])
        i+= 1
    '''
    labels = np.array(labels)
    return metas,labels


# %%
# train, test split dataset


# %%

def accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    return torch.sum(outputs == labels).item() / len(labels)



# %%

# %%
# model

vae = VQModel.from_pretrained(
            "CompVis/ldm-super-resolution-4x-openimages",
            #model_id,
              subfolder="vqvae",
            )
vae.eval()
scheduler = DDIMScheduler.from_pretrained(
            "CompVis/ldm-super-resolution-4x-openimages",
            #model_id, 
subfolder="scheduler")

metas,labels = build_metas("/root/autodl-fs/dtu2/Rectified/",1000)
dataset = TripletImageDataset(metas,labels,transforms.ToTensor(),vae,scheduler)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# %%

# %%



# %%
# train, test split dataset


# %%


class TripletVGG16(pl.LightningModule):
    def __init__(self,vae,scheduler):
        super(TripletVGG16, self).__init__()
        self.models = models.resnet18(pretrained=True)
        self.models.fc = nn.Linear(512,32)
        self.vae = vae.cuda()
        self.scheduler = scheduler
        self.time_proj = nn.Sequential(
            nn.Linear(1, 1024),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 32),
            nn.LeakyReLU(),
            

        )
         
    
        self.fc = nn.Linear(64,7)
        
        self.criteria = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
       
        # self.time_proj = nn.Sequential(
        #     nn.Linear(1, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 512),
            

        # )
   
    @torch.no_grad()
    def encoding_image(self,x):
        self.scheduler.set_timesteps(50, device=x.device)
        timesteps = self.scheduler.timesteps
        timestep = torch.randint(0, self.scheduler.num_train_timesteps,
                        (x.shape[0],), device=x.device).long()

        x = self.vae.encode(x).latents
        x = self.scheduler.add_noise(x, torch.randn_like(x), timestep)
        timestep = timestep.reshape(-1,1)
        return x,timestep.float()
    def configure_optimizers(self):
    
        
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=0.0001),
            #CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
            "monitor": "val_loss"
        }


        # scheduler = {
        #     #
        #     'scheduler': CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7),
        #     #ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=0.0001),
        #     #CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
        #     'interval': 'epoch',  # update the learning rate after each epoch
        #     'name': 'cosine_annealing_lr',
        #     "monitor": "val_loss"
        # }
        return  {'optimizer': optimizer, 'lr_scheduler': scheduler}
       
    def forward(self, x,timestep):
        
        time_emb = self.time_proj(timestep.float())
        x = self.models(x)
        x = torch.cat([x,time_emb],dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        #
        imgs,timestep = self.encoding_image(imgs)
        #imgs,timestep = self.encode_images(imgs)
        out = self(imgs,
                   timestep
                   )
        
        #loss = F.binary_cross_entropy_with_logits(out, labels)
        labels = labels.flatten()
        labels = nn.functional.one_hot(labels, num_classes=7).float()
        loss = self.criteria(out, labels)

        # round up loss to avoid valueerror for log
        loss = torch.round(loss * 1000) / 1000
        acc = accuracy(out, labels)
        metrics = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        #
        imgs,timestep = self.encoding_image(imgs)
        #imgs,timestep = self.encode_images(imgs)
        out = self(imgs,
                   timestep
                   )
        labels = labels.flatten()
        labels = nn.functional.one_hot(labels, num_classes=7).float()
        loss = self.criteria(out, labels)
       
        acc = accuracy(out, labels)

        loss = torch.round(loss * 1000) / 1000
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    @torch.no_grad()
    def prediction(self, imgs):
        
        out = self(imgs)
        return out
    


# %%

# %%





model = TripletVGG16(vae,scheduler)

epochs = 1000
logger = pl.loggers.TensorBoardLogger('/root/autodl-fs/tf-logs', name='classifier_single_frame')

callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/root/autodl-tmp/checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    save_last=True,
    mode='min',
)

trainer = Trainer(
    max_epochs=epochs,
    gpus=2,
    callbacks=[callback],
    precision = 16,
    strategy = "ddp",
    check_val_every_n_epoch=1,
    limit_val_batches=100,
    logger=logger,

    auto_scale_batch_size =True




    #resume_from_checkpoint="/root/autodl-tmp/checkpoints/model-epoch=000-val_loss=0.69.ckpt"
)
trainer.fit(model, train_loader, test_loader)






'''
def train(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler):
    for epoch in range(epochs):
        model.train()
        # define progress bar and each step print loss
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.cuda(), target.cuda()
            target = target.reshape(-1, 1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train_step', loss.item(), epoch*len(train_loader) + batch_idx)
        scheduler.step()
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            target = target.reshape(-1, 1)
            output = model.prediction(data)
            val_loss += criterion(output, target.float()).item()
            val_acc += accuracy(output, target)
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        print('Epoch: {} \tVal Loss: {:.6f} \tVal Acc: {:.6f}'.format(epoch, val_loss, val_acc))
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        # save model
        torch.save(model.state_dict(), '/root/autodl-tmp/checkpoints/model_classifier.pt'.format(epoch))


train(epochs, model, train_loader, test_loader, criterion, optimizer, scheduler)

'''


# %%



