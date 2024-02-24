import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main_fine_tune import instantiate_from_config
from taming.modules.util import SOSProvider
import sys
from einops import rearrange
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.depthloss import SL1Loss
sys.path.append('/root/autodl-tmp/project/dp_simple/CasMVSNet_pl')
from models.mvsnet import CascadeMVSNet
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from utils import *
from metrics import *  
from inplace_abn import ABN
import torchvision.transforms as T
from torchvision import utils as vutils

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float()
def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 #cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=True,

                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        # self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        # self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        self.depthmodel = CascadeMVSNet()
        self.transform = T.Compose([
                                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        self.unpreprocess = T.Compose([
                                T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                                ])
        self.depth_loss = SL1Loss()
        self.perception_loss = LPIPS()

 

    def decode_batch(self, batch):
        imgs = batch['imgs']
        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    # def init_cond_stage_from_ckpt(self, config):
    #     if config == "__is_first_stage__":
    #         print("Using first stage also as cond stage.")
    #         self.cond_stage_model = self.first_stage_model
    #     elif config == "__is_unconditional__" or self.be_unconditional:
    #         print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
    #               f"Prepending {self.sos_token} as a sos token.")
    #         self.be_unconditional = True
    #         self.cond_stage_key = self.first_stage_key
    #         self.cond_stage_model = SOSProvider(self.sos_token)
    #     else:
    #         model = instantiate_from_config(config)
    #         model = model.eval()
    #         model.train = disabled_train
    #         self.cond_stage_model = model

    def forward(self, x, c=None):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        # if not unconditioned:
        # _, c_indices = self.encode_to_c(c)

        # if self.training and self.pkeep < 1.0:
        #     mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
        #                                                  device=z_indices.device))
        #     mask = mask.round().to(dtype=torch.int64)
        #     r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
        #     a_indices = mask*z_indices+(1-mask)*r_indices
        # else:
        
        a_indices = z_indices

        # cz_indices = torch.cat((c_indices, a_indices), dim=1)
        cz_indices = a_indices
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        # logits = logits[:, c_indices.shape[1]-1:]

        return logits#, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def sample(self, x,# c,
                steps, temperature=0.5, sample=False, top_k=None,
               callback=lambda k: None):
        #x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        # assert not self.transformer.training
        #if self.pkeep <= 0.0:
        # one pass suffices since input is pure noise anyway
        assert len(x.shape)==2
        noise_shape = (x.shape[0], steps-1)
        #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
        #noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
        #x = torch.cat((x,noise),dim=1)
        logits, _ = self.transformer(x)
        # take all logits for now and scale by temp
        # gumbel softmax
        # logits = logits / temperature
        # # optionally crop probabilities to only the top k options
        # if top_k is not None:
        #     logits = self.top_k_logits(logits, top_k)
        # # apply softmax to convert to probabilities
        # probs = F.softmax(logits, dim=-1)
        logits = F.gumbel_softmax(logits, tau=temperature, hard=sample)
        # # sample from the distribution or take the most likely
        # if sample:
        #     shape = probs.shape
        #     probs = probs.reshape(shape[0]*shape[1],shape[2])
        #     ix = torch.multinomial(probs, num_samples=1)
        #     probs = probs.reshape(shape[0],shape[1],shape[2])
        #     ix = ix.reshape(shape[0],shape[1])
        # else:
        #     _, ix = torch.topk(probs, k=1, dim=-1)
        # cut off conditioning
        x = logits #[:, c.shape[1]-1:]
        # else:
        #     for k in range(steps):
        #         callback(k)
        #         assert x.size(1) <= block_size # make sure model can see conditioning
        #         x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        #         logits, _ = self.transformer(x_cond)
        #         # pluck the logits at the final step and scale by temperature
        #         logits = logits[:, -1, :] / temperature
        #         # optionally crop probabilities to only the top k options
        #         if top_k is not None:
        #             logits = self.top_k_logits(logits, top_k)
        #         # apply softmax to convert to probabilities
        #         probs = F.softmax(logits, dim=-1)
        #         # sample from the distribution or take the most likely
        #         if sample:
        #             ix = torch.multinomial(probs, num_samples=1)
        #         else:
        #             _, ix = torch.topk(probs, k=1, dim=-1)
        #         # append to the sequence and continue
        #         x = torch.cat((x, ix), dim=1)
        #     # cut off conditioning
        #     x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    # @torch.no_grad()
    # def encode_to_c(self, c):
    #     if self.downsample_cond_size > -1:
    #         c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
    #     quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
    #     if len(indices.shape) > 2:
    #         indices = indices.view(c.shape[0], -1)
    #     return quant_c, indices

   
    def decode_to_img(self, index, zshape,one_hot=True):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1,self.first_stage_model.quantize.re_embed), shape=bhwc,one_hot=one_hot)
        x = self.first_stage_model.decode(quant_z)
        x = x*0.5+0.5
       




        return x

   
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x = self.get_xc(batch, N)
        x = x.to(device=self.device)
        # c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        # quant_c, c_indices = self.encode_to_c(c)

        # create a "half"" sample
        z_start_indices = z_indices
        index_sample = self.sample(z_start_indices, # c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape,one_hot=True)



        # # sample
        # z_start_indices = z_indices[:, :0]
        # index_sample = self.sample(z_start_indices, # c_indices,
        #                            steps=z_indices.shape[1],
        #                            temperature=temperature if temperature is not None else 1.0,
        #                            sample=True,
        #                            top_k=top_k if top_k is not None else 100,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # # det sample
        # z_start_indices = z_indices[:, :0]
        # index_sample = self.sample(z_start_indices, # c_indices,
        #                            steps=z_indices.shape[1],
        #                            sample=False,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # # reconstruction
        # x_rec = self.decode_to_img(z_indices, quant_z.shape)

        # log["inputs"] = x
        # log["reconstructions"] = x_rec

        # if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
        #     figure_size = (x_rec.shape[2], x_rec.shape[3])
        #     dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
        #     label_for_category_no = dataset.get_textual_label_for_category_no
        #     plotter = dataset.conditional_builders[self.cond_stage_key].plot
        #     log["conditioning"] = torch.zeros_like(log["reconstructions"])
        #     for i in range(quant_c.shape[0]):
        #         log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
        #     log["conditioning_rec"] = log["conditioning"]
        # elif self.cond_stage_key != "image":
        #     cond_rec = self.cond_stage_model.decode(quant_c)
        #     if self.cond_stage_key == "segmentation":
        #         # get image from segmentation mask
        #         num_classes = cond_rec.shape[1]

        #         c = torch.argmax(c, dim=1, keepdim=True)
        #         c = F.one_hot(c, num_classes=num_classes)
        #         c = c.squeeze(1).permute(0, 3, 1, 2).float()
        #         c = self.cond_stage_model.to_rgb(c)

        #         cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
        #         cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
        #         cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
        #         cond_rec = self.cond_stage_model.to_rgb(cond_rec)
        #     log["conditioning_rec"] = cond_rec
        #     log["conditioning"] = c

        # log["samples_half"] = x_sample
        # log["samples_nopix"] = x_sample_nopix
        # log["samples_det"] = x_sample_det
        return x_sample

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        # c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            # c = c[:N]
            
        return x#, c

    # def shared_step(self, batch, batch_idx):
    #     x, c = self.get_xc(batch)
    #     logits, target = self(x, c)
    #     loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    #     return loss
    def content_loss(self, img_rec, img):
        img_rec = rearrange(img_rec, 'b v c h w -> b c h (v w)')
        img = rearrange(img, 'b v c h w -> b c h (v w)')
        return self.perception_loss(img_rec, img)
    def training_step(self, batch, batch_idx):
        # loss = self.shared_step(batch, batch_idx)
        # self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        b,v,c,h,w = imgs.shape
        imgs_input = rearrange(imgs, 'b v c h w -> b c h (v w)')
        batch['imgs'] = imgs_input
        img_rec = self.log_images(batch, temperature=0.5)
        img_rec = self.transform(img_rec)
        img_rec = rearrange(img_rec, 'b c h (v w) -> b v c h w', v=v)
       
        results = self.depthmodel(img_rec, proj_mats, init_depth_min, depth_interval)
        results_origin = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        depth_loss = self.depth_loss(results, depths, masks)

        content_loss = self.content_loss(img_rec, imgs)
        
        last_layer = self.first_stage_model.decoder.conv_out.weight
        if batch_idx%100 == 0:
            grad_content = torch.autograd.grad(content_loss, last_layer, retain_graph=True)[0]
            grad_depth = torch.autograd.grad(depth_loss, last_layer, retain_graph=True)[0]
            lambda_content = torch.norm(grad_depth) / (torch.norm(grad_content) + 1e-4)
            self.lambda_content = lambda_content

        content_loss = self.lambda_content * content_loss * 0.5

        loss = depth_loss + content_loss

        with torch.no_grad():
            losse_origin = self.depth_loss(results_origin, depths, masks)
        self.log("train/content_loss", content_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log("train/loss", depth_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_origin", losse_origin, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_diff", depth_loss/(losse_origin+1e-4), prog_bar=True, logger=True, on_step=True, on_epoch=True)

        log ={}
        with torch.no_grad():
            if batch_idx%10 == 0:
                try:
                    imgs_new =  img_rec
                    img1 = self.unpreprocess(imgs_new[0,0]).cpu() # batch 0, ref image
                    img2 = self.unpreprocess(imgs_new[0,1]).cpu() # batch 0, ref image
                    img3 = self.unpreprocess(imgs_new[0,2]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img1,img2,img3, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    epoch = self.current_epoch
                    vutils.save_image(stack,
                                       f"/root/autodl-tmp/taming-transformers/taming/results/train/{epoch}_pred_transformer_{batch_idx}.png", nrow=8)
                    
                    
                    img1 = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
                    img2 = self.unpreprocess(imgs[0,1]).cpu() # batch 0, ref image
                    img3 = self.unpreprocess(imgs[0,2]).cpu() # batch 0, ref image
                    # img_ = self.unpreprocess(imgs[0]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(results_origin['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(results_origin['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img1,img2,img3, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    # self.logger.experiment.add_images('train/image_GT_prob_old',
                    #                                 stack, self.global_step)
                    vutils.save_image(stack,
                                       f"/root/autodl-tmp/taming-transformers/taming/results/train/{epoch}_gt_transformer_{batch_idx}.png", nrow=8)
                    log['error'] =0
                   

                except Exception as e:
                    print(e)
                    log['error'] = 1
           
            mask = masks['level_0']
            depth_pred = results['depth_0']
            depth_old = results_origin['depth_0']
            depth_gt = depths['level_0']

            log['train/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            log['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            log['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            log['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
            log['train/abs_err_old'] = abs_error(depth_old, depth_gt, mask).mean()
            log['train/acc_1mm_old'] = acc_threshold(depth_old, depth_gt, mask, 1).mean()
            log['train/acc_2mm_old'] = acc_threshold(depth_old, depth_gt, mask, 2).mean()
            log['train/acc_4mm_old'] = acc_threshold(depth_old, depth_gt, mask, 4).mean()
            # the ratio of the loss
            log['train/abs_err_ratio'] = abs_err/(1e-10 + abs_error(depth_old, depth_gt, mask).mean())
            log['train/acc_1mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 1).mean())
            log['train/acc_2mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 2).mean())
            log['train/acc_4mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 4).mean())
            self.log_dict(log, on_epoch=True, on_step=True)

        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # loss = self.shared_step(batch, batch_idx)
        # self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        b,v,c,h,w = imgs.shape
        imgs_input = rearrange(imgs, 'b v c h w -> b c h (v w)')
        batch['imgs'] = imgs_input
        img_rec = self.log_images(batch, temperature=0.5)
        img_rec = self.transform(img_rec)
        img_rec = rearrange(img_rec, 'b c h (v w) -> b v c h w', v=v)
        results = self.depthmodel(img_rec, proj_mats, init_depth_min, depth_interval)
        results_origin = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        loss = self.depth_loss(results, depths, masks)
        
        losse_origin = self.depth_loss(results_origin, depths, masks)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_origin", losse_origin, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_diff", loss/(losse_origin+1e-4), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log ={}
        if batch_idx%10 == 0:
            try:
                imgs_new =  img_rec
                img1 = self.unpreprocess(imgs_new[0,0]).cpu() # batch 0, ref image
                img2 = self.unpreprocess(imgs_new[0,1]).cpu() # batch 0, ref image
                img3 = self.unpreprocess(imgs_new[0,2]).cpu() # batch 0, ref image
                depth_gt_ = visualize_depth(depths['level_0'][0])
                depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                stack = torch.stack([img1,img2,img3, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                epoch = self.current_epoch
                vutils.save_image(stack,
                                    f"/root/autodl-tmp/taming-transformers/taming/results/train/{epoch}_pred_transformer_{batch_idx}.png", nrow=8)
                
                
                img1 = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
                img2 = self.unpreprocess(imgs[0,1]).cpu() # batch 0, ref image
                img3 = self.unpreprocess(imgs[0,2]).cpu() # batch 0, ref image
                # img_ = self.unpreprocess(imgs[0]).cpu() # batch 0, ref image
                depth_gt_ = visualize_depth(depths['level_0'][0])
                depth_pred_ = visualize_depth(results_origin['depth_0'][0]*masks['level_0'][0])
                prob = visualize_prob(results_origin['confidence_0'][0]*masks['level_0'][0])
                stack = torch.stack([img1,img2,img3, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                # self.logger.experiment.add_images('train/image_GT_prob_old',
                #                                 stack, self.global_step)
                vutils.save_image(stack,
                                    f"/root/autodl-tmp/taming-transformers/taming/results/val/{epoch}_val_gt_transformer_{batch_idx}.png", nrow=8)
                log['error'] =0
                

            except:

                log['error'] = 1
            
            mask = masks['level_0']
            depth_pred = results['depth_0']
            depth_old = results_origin['depth_0']
            depth_gt = depths['level_0']

            log['val/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            log['val/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            log['val/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            log['val/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
            log['val/abs_err_old'] = abs_error(depth_old, depth_gt, mask).mean()
            log['val/acc_1mm_old'] = acc_threshold(depth_old, depth_gt, mask, 1).mean()
            log['val/acc_2mm_old'] = acc_threshold(depth_old, depth_gt, mask, 2).mean()
            log['val/acc_4mm_old'] = acc_threshold(depth_old, depth_gt, mask, 4).mean()
            # the ratio of the loss
            log['val/abs_err_ratio'] = abs_err/(1e-10 + abs_error(depth_old, depth_gt, mask).mean())
            log['val/acc_1mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 1).mean())
            log['val/acc_2mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 2).mean())
            log['val/acc_4mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 4).mean())
            self.log_dict(log, on_epoch=True, on_step=True)
        return loss
    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
