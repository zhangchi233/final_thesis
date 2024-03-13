from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import yaml
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateMonitor

import sys

sys.path.append("/root/autodl-tmp/project")
sys.path.append("/root/autodl-tmp/taming-transformers")
from framework.lora.dataset.dtu import DTUDataset

#from framwork.inpainting.lighting_uncliponly import PanoOutpaintGenerator
from framwork.ldm.lightning_ldm import PanoOutpaintGenerator 


class CallbackLogger(Callback):
    def __init__(self):
        super().__init__()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        dark_imgs = outputs["dark_imgs"]
        gt_imgs = outputs["gt_imgs"]
        pred_imgs = outputs["pred_imgs"]
        mask = outputs["mask"].repeat(1,3,1,1)
        ## use logger to log images
        dark_imgs = np.transpose(dark_imgs,(0,3,1,2))
        gt_imgs = np.transpose(gt_imgs,(0,3,1,2))
        pred_imgs = np.transpose(pred_imgs,(0,3,1,2))
       
        
        
        


        trainer.logger.experiment.add_images("dark_imgs", dark_imgs, global_step=trainer.global_step)
        trainer.logger.experiment.add_images("gt_imgs", gt_imgs, global_step=trainer.global_step)
        trainer.logger.experiment.add_images("pred_imgs", pred_imgs, global_step=trainer.global_step)
        #trainer.logger.experiment.add_images("mask", mask, global_step=trainer.global_step)
        print("add images")


if __name__ == "__main__":
    import torch
    data_train = DTUDataset(root_dir="/root/autodl-fs/dtu2/",
                      split="train",
                     
                      # train_test=True,
                      
                      )
    class datawarper:
        def __init__(self,dataset):
            self.dataset=dataset
            self.split = dataset.split
            

        def __getitem__(self,index):
            try:
                if index == 0 and self.split != "train":
                    
                    import cv2
                    import torchvision.transforms as T
                    path1 = "/root/autodl-tmp/frames/case_0/frame_235.jpg"
                    path2 = "/root/autodl-tmp/frames/case_0/frame_234.jpg"
                    path3 = "/root/autodl-tmp/frames/case_0/frame_236.jpg"
                    img1 = cv2.imread(path1)
                    img2 = cv2.imread(path2)
                    img3 = cv2.imread(path3)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
                    img1 = T.ToTensor()(img1).unsqueeze(0).cuda()*2-1
                    img2 = T.ToTensor()(img2).unsqueeze(0).cuda()*2-1
                    img3 = T.ToTensor()(img3).unsqueeze(0).cuda()*2-1
                    dark_imgs = torch.cat([img1, img2, img3])
                    imgs = torch.cat([img2, img2, img3])
                    sample = self.dataset[index]
                    sample["dark_imgs"] = dark_imgs.cuda()
                    sample["imgs"] = imgs.cuda()
                    sample["mask"] = torch.zeros(1,1,512,512).cuda()
                    
                    return sample
                return self.dataset[index]
            except Exception as e:
                print(e)
                print("index",index)
                return self.__getitem__(index+1)
        def __len__(self):
            return len(self.dataset)
    
    datasets_dtu = datawarper(data_train)
    train_loader = torch.utils.data.DataLoader(
        datasets_dtu, batch_size=16, shuffle=True, num_workers=0,
      )

    val_data = DTUDataset(root_dir="/root/autodl-fs/dtu2/",split="val",img_wh=(512, 512))
    data_val = datawarper(val_data)
    # only sample 10 batches for validation
    

    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=1, shuffle=True, num_workers=0)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')


    
    config = yaml.load(open("/root/autodl-tmp/project/configs/config_sr_ldm.yaml", "r"), Loader=yaml.FullLoader)

    

    # Step 2: Create the Trainer
    trainer_config = config["Trainer"].get("trainer", {})
    checkpoint_config = config["Trainer"].get("checkpoint", {})
   

    # Create callbacks
   
    
    model = PanoOutpaintGenerator(config)
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    callback = CallbackLogger()
    callbacks = [checkpoint_callback,callback,lr_monitor]




        # Create logger
    logger = pl.loggers.TensorBoardLogger(**config["Trainer"].get("logger", {}))

    trainer = pl.Trainer(
            **trainer_config,
            callbacks=callbacks,
            logger=logger, max_epochs=400,limit_val_batches=10,
            #resume_from_checkpoint="/root/autodl-tmp/project/checkpoints/last_model-v2.ckpt"
            
    )
    # set batch size automatically
    trainer.fit(model, train_loader, val_loader)
