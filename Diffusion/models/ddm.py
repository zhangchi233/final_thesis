import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from metrics.metric import abs_error, acc_threshold
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from tensorboardX import SummaryWriter
import torchvision.transforms as T
from models.losses import SL1Loss
from models.mvsnet import CascadeMVSNet
from models.modules import load_ckpt
from inplace_abn import ABN
from einops import rearrange
import random
def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        
        self.device = self.config.device
        self.high_enhance0 = HFRM(in_channels=3, out_channels=64,use_lora=config.model.use_lora,ranks=config.model.hfrm_ranks)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64,use_lora=config.model.use_lora,ranks=config.model.hfrm_ranks)
        self.Unet = DiffusionUNet(config)
        


        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]
    
    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]
    
    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, :3, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]
            
            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())
            denoise_LL_LL = self.sample_training(input_LL_LL, b)

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        
        if config.logger is not None:
            exp_name = config.logger.exp_name
            logger_path = config.logger.logger_path
            # calculate the number of experiments with the same name
            num=0
            if os.path.exists(os.path.join(logger_path, exp_name)):
                num = 0
                while os.path.exists(os.path.join(logger_path, f'{exp_name}_{num}')):
                    num += 1

                exp_name = exp_name + f'_{num}'
            else:
                exp_name = exp_name
            
            self.logger = SummaryWriter(log_dir=os.path.join(logger_path, exp_name))
        
        if config.model.use_depth:
            self.optimizer = None
            self.scheduler = None
            self.depth_model = CascadeMVSNet(n_depths=[8,32,48],
                        interval_ratios=[1.0,2.0,4.0],
                        num_groups=1,
                        
                        norm_act=ABN).cuda()
            #load_ckpt(self.depth_model, '/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')
            self.depth_transform = T.Compose([
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225]),
                                       ])
            print("load depth model: ", self.depth_model)
            self.depth_optimizer, self.depth_scheduler = utils.optimize.get_optimizer(self.config, self.depth_model.parameters())
            self.depth_loss = SL1Loss()
            if self.config.model.training_mode=="include_decoder":
                
                trainable_params =[]
                
                for name, param in self.model.named_parameters():
                    if "high_enhance" in name:
                        trainable_params.append(param)
                    elif "Unet.up.3" in name:
                        trainable_params.append(param)
                    else:
                        param.requires_grad_(False)

                
                
                self.depth_optimizer, self.depth_scheduler = utils.optimize.get_optimizer(self.config, list(self.depth_model.parameters())+\
                                                                                          trainable_params)
            elif self.config.model.training_mode=="only_mvs":
                for param in self.model.module.parameters():
                    
                    param.requires_grad = False    
            elif self.config.model.training_mode=="full_model":
                trainable_params =list(self.model.parameters())
                self.depth_optimizer, self.depth_scheduler = utils.optimize.get_optimizer(self.config, list(self.depth_model.parameters())+\
                                                                                          trainable_params)
            elif self.config.model.training_mode=="including_image_noise":
                self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
                self.not_train_diffusion = False
            else:

                self.optimizer = None
                self.scheduler = None
                
        else:
            self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
            self.depth_scheduler=None
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))
    def decode_batch(self, batch):
       
        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']

        return proj_mats, depths, masks, init_depth_min, depth_interval
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        train_loader = train_loader
        val_loader = val_loader

        

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            from tqdm import tqdm

            train_loader = tqdm(train_loader)
            self.model.train()
            for i, batch in enumerate(train_loader):
                # print(x.shape,x.min(),x.max())
                img_ori = batch["imgs_train"]
                img_gt = batch["imgs"]
                x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
               
                x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x

                y = batch["scan_vid"]
                
                data_time += time.time() - data_start
                
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)
                
                

                if self.config.model.use_depth: #and ((random.random() < 0.35) or (self.step % 10 == 0)):
                    proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
                    

                    imgs = output["pred_x"]
                    
                    # reshape to (B, N, C, H, W)
                    proj_mats = proj_mats.to(self.device)
                    init_depth_min = init_depth_min.to(self.device)
                    depth_interval = depth_interval.to(self.device)
                    for level in depths:
                        depths[level] = depths[level].to(self.device)
                        masks[level] = masks[level].to(self.device)
                    for level in masks:
                        masks[level] = masks[level].to(self.device)
                    

                    b, v = img_ori.shape[0], img_ori.shape[1]
                    imgs = rearrange(imgs, '(b v) c h w -> b v c h w', b=b, v=v) 
                    if self.config.model.debug_mvs_only:
                        imgs = img_ori.to(self.device)
                    imgs = self.depth_transform(imgs)
                   

                    results = self.depth_model(imgs, proj_mats, init_depth_min, depth_interval)
                    depth_loss = self.depth_loss(results, depths, masks)
                    self.depth_optimizer.zero_grad()
                    #self.optimizer.zero_grad()
                    depth_loss.backward()
                    self.depth_optimizer.step()
                    #self.optimizer.step()
                    
                    log = {}
                    depth_pred = results['depth_0']
                    depth_gt = depths['level_0']
                    mask = masks['level_0']
                    # self.logger.add_scalars['train/abs_err'] =  abs_error(depth_pred, depth_gt, mask).mean()
                    # self.logger.add_scalars['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
                    # self.logger.add_scalars['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
                    # self.logger.add_scalars['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
                    # self.logger.add_scalars["train/depth_loss"] = depth_loss.item()
                    
                    self.logger.add_scalar("train/abs_err", abs_error(depth_pred, depth_gt, mask).mean(), self.step)
                    self.logger.add_scalar("train/acc_1mm", acc_threshold(depth_pred, depth_gt, mask, 1).mean(), self.step)
                    self.logger.add_scalar("train/acc_2mm", acc_threshold(depth_pred, depth_gt, mask, 2).mean(), self.step)
                    self.logger.add_scalar("train/acc_4mm", acc_threshold(depth_pred, depth_gt, mask, 4).mean(), self.step)
                    self.logger.add_scalar("train/depth_loss", depth_loss.item(), self.step)




                    imgs = img_gt.to(self.device)
                    
                    imgs = self.depth_transform(imgs)
                    ori_results = self.depth_model(imgs, proj_mats, init_depth_min, depth_interval)
                    ori_loss = self.depth_loss(ori_results, depths, masks)
                    self.logger.add_scalar("train/depth_ori", ori_loss.item(), self.step)
                    self.logger.add_scalar("train/depth_loss_ratio", (depth_loss/ori_loss), self.step)

                    


                    


                    

                            


                if not self.config.model.not_train_diffusion:
                    noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                

                    loss = noise_loss + photo_loss + frequency_loss
                    depth_loss = 0
                    ori_loss = 1
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                 
                
                
                if self.step % 10 == 0:
                   
                    with torch.no_grad():
                        batch = next(iter(val_loader))
                        img_ori = batch["imgs_train"]
                        img_gt = batch["imgs"]
                        x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
                        proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
                        proj_mats = proj_mats.to(self.device)
                        init_depth_min = init_depth_min.to(self.device)
                        depth_interval = depth_interval.to(self.device)
                        x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x

                        y = batch["scan_vid"]
                        
                       

                        x = x.to(self.device)

                        output = self.model(x)
                        noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                

                        loss = noise_loss + photo_loss + frequency_loss
                       
                    if hasattr(self, "logger"):
                        self.logger.add_scalar("train/noise_loss", noise_loss.item(), self.step)
                        self.logger.add_scalar("train/photo_loss", photo_loss.item(), self.step)
                        self.logger.add_scalar("train/frequency_loss", frequency_loss.item(), self.step)
                        self.logger.add_scalar("train/total_loss", loss.item(), self.step,)
                        
                        

                    train_loader.set_description_str("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}, depth_loss_ratio:{:.4f}, depth_loss:{:.4f}, depth_ori: {:.4f}".format(self.step, self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.depth_scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item(), (depth_loss/ori_loss)
                                                         ,depth_loss,ori_loss),
                                                         refresh=True)
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 or self.step ==1:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                                                   'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))
                    self.model.train()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.depth_scheduler is not None:
                self.depth_scheduler.step()
    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_LL))

        # =============photo loss==================
        
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")

            # randomly sample a 10 batch of validation images
            val_loader_iter = iter(val_loader)
            for i in range(10):
                batch = next(val_loader_iter)
                
                img_input = batch["imgs_train"]
                img_gt = batch["imgs"]

                x = torch.cat([img_input, img_gt], dim=2 if img_input.ndim == 5 else 1)

                x = img_input.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                
                y = batch["scan_vid"]
               

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                pred_x = pred_x.reshape(img_input.shape)

                
                if self.config.model.use_depth: #and ((random.random() < 0.35) or (self.step % 10 == 0)):
                    proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
                    
                    imgs = out["pred_x"]
                    # reshape to (B, N, C, H, W)
                    proj_mats = proj_mats.to(self.device)
                    init_depth_min = init_depth_min.to(self.device)
                    depth_interval = depth_interval.to(self.device)
                    for level in depths:
                        depths[level] = depths[level].to(self.device)
                        masks[level] = masks[level].to(self.device)
                    for level in masks:
                        masks[level] = masks[level].to(self.device)
                    

                    b, v = img_gt.shape[0], img_gt.shape[1]
                    imgs = rearrange(imgs, '(b v) c h w -> b v c h w', b=b, v=v) 
                    
                    imgs = self.depth_transform(imgs)
                   

                    results = self.depth_model(imgs, proj_mats, init_depth_min, depth_interval)
                    depth_loss = self.depth_loss(results, depths, masks)
                    
                    #self.optimizer.step()
                    
                    log = {}
                    depth_pred = results['depth_0']
                    depth_gt = depths['level_0']
                    mask = masks['level_0']
                    # self.logger.add_scalars['train/abs_err'] =  abs_error(depth_pred, depth_gt, mask).mean()
                    # self.logger.add_scalars['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
                    # self.logger.add_scalars['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
                    # self.logger.add_scalars['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
                    # self.logger.add_scalars["train/depth_loss"] = depth_loss.item()
                    
                    



                    imgs = img_input.to(self.device)
                    
                    imgs = self.depth_transform(imgs)
                    ori_results = self.depth_model(imgs, proj_mats, init_depth_min, depth_interval)
                    ori_loss = self.depth_loss(ori_results, depths, masks)
                    
               
                if hasattr(self, "logger"):
                    #
                    if len(img_input.shape) == 5:
                        img_input = rearrange(img_input, 'b v c h w ->c h (b v w)')
                        img_gt = rearrange(img_gt, 'b v c h w ->c h (b v w)')
                        pred_x = rearrange(pred_x, 'b v c h w ->c h (b v w)')
                    else:
                        img_input = rearrange(img_input, 'v c h w ->c h (v w)')
                        img_gt = rearrange(img_gt, 'v c h w ->c h (v w)')
                        pred_x = rearrange(pred_x, 'v c h w ->c h (v w)')
                    
                    self.logger.add_image("val/input", img_input, step)
                    self.logger.add_image("val/gt", img_gt, step)
                    self.logger.add_image("val/pred", pred_x, step)
                    self.logger.add_scalar("val/abs_err", abs_error(depth_pred, depth_gt, mask).mean(), self.step)
                    self.logger.add_scalar("val/acc_1mm", acc_threshold(depth_pred, depth_gt, mask, 1).mean(), self.step)
                    self.logger.add_scalar("val/acc_2mm", acc_threshold(depth_pred, depth_gt, mask, 2).mean(), self.step)
                    self.logger.add_scalar("val/acc_4mm", acc_threshold(depth_pred, depth_gt, mask, 4).mean(), self.step)
                    self.logger.add_scalar("val/depth_loss", depth_loss.item(), self.step)
                    self.logger.add_scalar("val/depth_ori", ori_loss.item(), self.step)
                    self.logger.add_scalar("val/depth_loss_ratio", (depth_loss/ori_loss), self.step)
                
                saves = torch.stack([img_input, img_gt, pred_x.cpu()])

                utils.logging.save_image(saves, os.path.join(image_folder,self.config.logger.exp_name, str(step), f"{y[0]}_{i}.png"))




