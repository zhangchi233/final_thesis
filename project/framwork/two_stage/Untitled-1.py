# %%
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# %%
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# %%
class TripletImageDataset(Dataset):
    def __init__(self, img_triplets, labels, transform=None):
        """
        img_triplets: List of tuples (img_path1, img_path2, img_path3)
        labels: List of binary labels
        transform: torchvision transforms for preprocessing
        """
        self.img_triplets = img_triplets
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_triplets)

    def __getitem__(self, idx):
        img1  = self.img_triplets[idx]
        label = self.labels[idx]

        # Load images and apply transformations
        try:
            imgs = transforms.functional.pil_to_tensor(Image.open(img1).resize((128,128)))/127.5-1
           
        except Exception as e:
            print("Error loading image",e)
            print(self.img_triplets[idx])
            return self.__getitem__(idx + 1)
        
        
        
        

       

        return imgs, label

# %%
import os
import numpy as np
import tqdm
def build_metas(path,examples):
    files = os.listdir(path)
    
    #img_format
    #define positive examples
    labels = []
    metas = []
    i=0

    
    for file in files:

        for image in os.listdir(os.path.join(path,file)):
            if image.endswith(".png"):
               label = image.split("_")[2]
               label = int(label)
               labels.append(label)
               metas.append((os.path.join(path,file,image)))       
    #define negative examples
   

    return metas,labels
metas,labels = build_metas("/root/autodl-fs/dtu2/Rectified/",1000)
dataset = TripletImageDataset(metas,labels,transforms.ToTensor())
img,label = dataset[0]
print(img.shape)
print(label)

# %%
import torch.nn.functional as F
def accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)
    labels = labels.reshape(-1)
    outputs = outputs.reshape(-1)
    return torch.sum(outputs == labels).item() / len(labels)
class TripletVGG16(nn.Module):
    def __init__(self):
        super(TripletVGG16, self).__init__()
        # Load pre-trained VGG-16 model
        self.vgg16 = models.resnet18(pretrained=True)
        # Remove the classification layer
        self.vgg16 = nn.Sequential(*list(self.vgg16.children())[:-1])
        # Freeze the parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg16[-3:].parameters():
            param.requires_grad = True
       #self.time_proj = nn.Linear(1, 512)
        # Add custom classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Pass each image through VGG-16
        x1 = x
        
        x1 = self.vgg16(x1)
        
       
        #time_proj = self.time_proj(timestep)

        # Flatten and concatenate features
        x1 = x1.view(x1.size(0), -1)
        
        

        # Classification
        out = self.classifier(x1)
        return out
    def training_step(self, batch):
        imgs, labels = batch
        out = self(imgs)
        loss = F.binary_cross_entropy(out, labels)
        return loss
    @torch.no_grad()
    def validation_step(self, batch):
        imgs, labels = batch
        out = self(imgs)
        loss = F.binary_cross_entropy(out, labels)
        acc = accuracy(out, labels)
        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    @torch.no_grad()
    def prediction(self, imgs):
        
        out = self(imgs)
        return out

# %%


# %%
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
# import tensorboard 
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/root/tf-logs/classifier')

# %%

# %%
model = TripletVGG16().cuda()
model.load_state_dict(torch.load('/root/autodl-tmp/checkpoints/model_classifier_2.pt'))

# %%
from einops import rearrange

# %%
# train, test split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=521, shuffle=True,num_workers=8)
len(train_loader), len(test_loader)

# %%
from tqdm import tqdm
# define loss function
criterion = nn.CrossEntropyLoss()
# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# define scheduler, use cosine annealing, decay the learning rate to 1e-6
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
epochs = 10
def train(epochs, model, train_loader, test_loader, criterion, optimizer, lr_scheduler):
    for epoch in range(epochs):
        model.train()
        # define progress bar and each step print loss
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
           
            data, target = data.cuda(), target.cuda()
               
               
                
            
            optimizer.zero_grad()
               
               
              
                
                
            output = model(data)
            print(output.shape,target.shape)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            
            acc = accuracy(output, target)
            print(loss.item(),acc)
            

            # writer.add_scalar('Loss/train_step', loss.item(), epoch*len(train_loader) + batch_idx)
        lr_scheduler.step()
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data, target = data.cuda(), target.cuda()
                
                

                

                output = model(data)
                print(output.shape,target.shape)
                val_loss += criterion(output,target).item()
                val_acc += accuracy(output, target)
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        print('Epoch: {} \tVal Loss: {:.6f} \tVal Acc: {:.6f}'.format(epoch, val_loss, val_acc))
        # writer.add_scalar('Loss/train', loss.item(), epoch)
        # writer.add_scalar('Loss/test', val_loss, epoch)
        # writer.add_scalar('Accuracy/test', val_acc, epoch)
        # save model
        torch.save(model.state_dict(), '/root/autodl-tmp/checkpoints/model_classifier_2.pt'.format(epoch))


train(100, model, train_loader, test_loader, criterion, optimizer, lr_scheduler)


