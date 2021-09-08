import os
import shutil
import argparse

import torch
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import utils_p
from dataloader import D2PDataset
from my_models import UNet


parser = argparse.ArgumentParser()
parser.add_argument('--model_trans_path', type=str, default='./data_saved/UNet_trans.pth',
                    help='the path of digital-to-physical UNet transformation model')
parser.add_argument('--dataroot', type=str, default='./data/d2p/test', help='images path')
parser.add_argument('--save_dir', type=str, default='./data/predict_right_images', help='images path to save')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. Only support 1 for now')
parser.add_argument('--gpu', type=int, default=0, help='gpu to use')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)


# Todo: Image transformations
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_ori = transforms.Compose([
    transforms.Resize(round(224 * 1.05)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

tf_inv = transforms.Compose([
    transforms.Normalize(
        mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
        std=[1 / std[0], 1 / std[1], 1 / std[2]],
    ),
])


loader = DataLoader(
    D2PDataset(opt.dataroot, transforms_=tf_ori),
    batch_size=opt.batch_size,
    shuffle=False
)


# TODO: loading classifier model
print('==>Loading classifier model ResNet50...')
classifier = models.resnet50(pretrained=True).cuda().eval()


# TODO: loading transformation model
norm_min, norm_max = utils_p.get_normal_min_max()
model_trans = UNet(norm_min, norm_max, use_clamp=True)
print('==>Loading model trans...')
ckpt = torch.load(opt.model_trans_path, map_location='cuda:0')
model_trans = ckpt['model'].cuda()
model_trans.eval()


correct, total = 0, 0
wrong_correct_path_list = []
right_correct_path_list = []
for batch_data in loader:
    images_D = batch_data['D'].cuda()
    images_P = batch_data['P'].cuda()
    labels = batch_data['label'].cuda()
    images_D_path = batch_data['d_path']
    images_P_path = batch_data['p_path']

    images_predict = model_trans(images_D)

    outputs = classifier(images_predict)
    _, predicted = torch.max(outputs.data, 1)

    total += 1
    if predicted == labels:
        correct += 1
        right_correct_path_list.append([images_D_path[0], images_P_path[0]])
    else:
        wrong_correct_path_list.append([images_D_path[0], images_P_path[0]])

acc = correct / total
print('acc: {:.6f} \t {}/{}'.format(acc, correct, total))


digital_save_path = os.path.join(opt.save_dir, 'digital')
physical_save_path = os.path.join(opt.save_dir, 'physical')
utils_p.is_exists(digital_save_path)
utils_p.is_exists(physical_save_path)

for path in right_correct_path_list:
    d_path = path[0]
    p_path = path[1]

    src = d_path
    image_name = src.split('/')[-1]
    dst = os.path.join(digital_save_path, image_name)
    shutil.copyfile(src, dst)

    src = p_path
    image_name = src.split('/')[-1]
    dst = os.path.join(physical_save_path, image_name)
    shutil.copyfile(src, dst)








