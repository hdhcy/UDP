import os
import json
import argparse
from tqdm import tqdm

import torch
from torch import optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils_p
from my_models import UNet
from dataloader import im_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_trans_path', type=str, default='./data_saved/UNet_trans.pth',
                    help='the path of digital-to-physical UNet transformation model')
parser.add_argument('--dataroot', type=str, default='./data/predict_right_images/digital',
                    help='path to images to attack')
parser.add_argument('--res_save_path', type=str, default='./data_saved/UDP_res',
                    help='path to save attack results')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. Only support 1 for now')
parser.add_argument('--adv_label', type=int, default=521, help='target attack category label')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epsilon', type=int, default=16, help='upper limit of perturbation')
parser.add_argument('--num_iter', type=int, default=20, help='the number of iterations of the training perturbation')
parser.add_argument('--data_size', type=int, default=5000, help='Total transformations per EOT-S')
parser.add_argument('--data_batch_size', type=int, default=25, help='batch size for each gradient descent')
parser.add_argument('--start_index', type=int, default=0, help='start index')
parser.add_argument('--aerf', type=float, default=1., help=' adversarial intensity between digital and physical')
parser.add_argument('--gpu', type=int, default=0, help='gpu to use')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
aerf = opt.aerf
start_index = opt.start_index


# TODO: loading classifier model
print('==>Loading classifier model ResNet50...')
classifier = models.resnet50(pretrained=True).cuda().eval()
criterion = torch.nn.CrossEntropyLoss()
class_idx = json.load(open("data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


# TODO: loading transformation model
norm_min, norm_max = utils_p.get_normal_min_max()
model_trans = UNet(norm_min, norm_max, use_clamp=True)
print('==>Loading model trans...')
ckpt = torch.load(opt.model_trans_path, map_location='cuda:0')
model_trans = ckpt['model'].cuda()
model_trans.eval()


# Todo: Image transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

trf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trf_nor = transforms.Compose([
    transforms.Normalize(mean, std)
])

trf_nor_inv = transforms.Compose([
    transforms.Normalize(
        mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
        std=[1 / std[0], 1 / std[1], 1 / std[2]],
    ),
])

dataset = im_dataset(opt.dataroot)
dataset_loader = DataLoader(dataset, batch_size=opt.batch_size)


adv_label = torch.tensor(opt.adv_label, dtype=torch.long)
adv_label = torch.unsqueeze(adv_label, 0).cuda()

lr_rate = opt.lr

epsilon = opt.epsilon / 255.
print('epsilon: {:.4f}'.format(epsilon))

num_iter = opt.num_iter
data_size, batch_size = opt.data_size, opt.data_batch_size
assert data_size >= batch_size, 'data_size must be greater than batch_size'

for index, (image, file_name) in enumerate(dataset_loader):
    if index >= start_index:
        print('--------' * 4, index, '--------' * 4)
        image_tensor = image.cuda()
        utils_p.image_show(image_tensor.squeeze(0))

        perturb = torch.zeros_like(image_tensor)
        optimizer = optim.Adam([perturb.requires_grad_()], lr=lr_rate, betas=(0.9, 0.999))

        save_dir = '{}/{}_index_{}'.format(opt.res_save_path, file_name[0], index)
        utils_p.is_exists(save_dir)

        for epoch in range(num_iter):

            # train a perturbation
            total_train, loss_total = 0, 0
            for _ in tqdm(range(int(data_size / batch_size) + 1)):
                adv_image_tensor = utils_p.get_adv_image_tensor(image_tensor, perturb, epsilon)
                loss = utils_p.compute_eot_loss_normal(adv_image_tensor, classifier, model_trans, criterion,
                                                       adv_label, trf_nor, batch_size, aerf=aerf)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train += 1
                loss_total += loss.item()

            loss_avg = loss_total / total_train
            print('epoch: [{}/{}] loss: {:.6f}'.format(epoch + 1, num_iter, loss_avg))

            # test
            adv_D_total, adv_P_total = 0, 0
            adv_con_D_total, adv_con_P_total = 0, 0
            total_test = 0
            with torch.no_grad():
                adv_image_tensor = utils_p.get_adv_image_tensor(image_tensor, perturb, epsilon)
                for _ in tqdm(range(int(data_size / batch_size / 4) + 1)):
                    eot_images = utils_p.get_eot_images(adv_image_tensor, batch_size)
                    for i in range(eot_images.shape[0]):
                        adv_image_D = trf_nor(eot_images[i])
                        adv_image_P = model_trans(adv_image_D)

                        adv_label_D, adv_conf_D, _ = utils_p.get_image_predict(classifier, adv_image_D, idx2label)
                        adv_label_P, adv_conf_P, _ = utils_p.get_image_predict(classifier, adv_image_P, idx2label)
                        if adv_label_D == adv_label.item():
                            adv_D_total += 1
                            adv_con_D_total += adv_conf_D
                        if adv_label_P == adv_label.item():
                            adv_P_total += 1
                            adv_con_P_total += adv_conf_P
                    total_test += eot_images.shape[0]

                adv_D_acc = adv_D_total / total_test
                adv_P_acc = adv_P_total / total_test
                adv_D_con = adv_con_D_total / (adv_D_total + 1e-8)
                adv_P_con = adv_con_P_total / (adv_P_total + 1e-8)
                print('adv digital acc: {:.4f} {}/{} con: {:.6f} '
                      'adv physical acc: {:.4f} {}/{} con: {:.6f}'
                      .format(adv_D_acc, adv_D_total, total_test, adv_D_con,
                              adv_P_acc, adv_P_total, total_test, adv_P_con))

            state = {
                'aref': aerf,
                'epoch': epoch,
                'lr_rate': lr_rate,
                'num_iter': num_iter,
                'save_dir': save_dir,
                'adv_label': adv_label.data,
                'ori_image': image_tensor.data,
                'adv_image': adv_image_tensor.data,
                'data_size': data_size,
                'batch_size': batch_size,
                'train': {
                    'loss': loss_avg,
                    'total_train': total_train,
                },
                'test': {
                    'adv_D_acc': adv_D_acc,
                    'adv_D_con': adv_D_con,
                    'adv_P_acc': adv_P_acc,
                    'adv_P_con': adv_P_con,
                    'total_test': total_test,
                }
            }
            torch.save(state, '{}/adv_phy_{}.pth'
                       .format(save_dir, str(epoch + 1).zfill(2)))

            utils_p.image_show(adv_image_tensor.data.squeeze(0), title='epoch: {}'.format(epoch + 1))

            image_save_path = 'adv_phy_{}.png'.format(str(epoch + 1).zfill(2))
            save_image(adv_image_tensor.data, os.path.join(save_dir, image_save_path))
        print()
        print()

