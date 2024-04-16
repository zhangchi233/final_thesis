import torch
from pytorch_msssim import ssim
from torch import nn
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
class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        return loss
def estimation_loss(x, output,device="cuda"):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(device)
        TV_loss = TVLoss()
        l2_loss = torch.nn.MSELoss()
        l1_loss = torch.nn.L1Loss()

        # =============noise loss==================
        noise_loss = l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * (l2_loss(input_high0, gt_high0) +
                                l2_loss(input_high1, gt_high1) +
                                l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (TV_loss(input_high0) +
                                 TV_loss(input_high1) +
                                 TV_loss(pred_LL))

        # =============photo loss==================
        
        content_loss = l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss
loss_dict = {'sl1': SL1Loss}