import torch

import utils_p
from my_models import UNet

norm_min, norm_max = utils_p.get_normal_min_max()
model_trans = UNet(norm_min, norm_max, use_clamp=True)
model_trans_save_path = './data_saved/UNet_trans.pth'
print('==>Loading model trans...')
ckpt = torch.load(model_trans_save_path, map_location='cuda:0')
print(ckpt.keys())

model_trans = ckpt['model'].cuda()
print('model_trans epoch: {}'.format(ckpt['epoch']))
model_trans.eval()

