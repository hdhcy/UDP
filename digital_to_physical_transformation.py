import os
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import utils_p
from dataloader import D2PDataset
from my_models import LambdaLR, UNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./data/d2p', help='digital and physical images path')
parser.add_argument('--save_path', type=str, default='./data_saved/UNet_trans.pth',
                    help='path to images to save model')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--n_epochs', type=int, default=400, help='the number of epochs of training')
parser.add_argument('--offset', type=int, default=0, help='the epoch to start training from')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
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

# Todo: dataloader
train_root = os.path.join(opt.dataroot, 'train')
print(train_root)
train_loader = DataLoader(
    D2PDataset(train_root, transforms_=tf_ori),
    batch_size=opt.batch_size,
    shuffle=True
)

test_root = os.path.join(opt.dataroot, 'test')
test_loader = DataLoader(
    D2PDataset(test_root, transforms_=tf_ori),
    batch_size=opt.batch_size,
    shuffle=False
)

# Todo: loading transformation model
norm_min, norm_max = utils_p.get_normal_min_max()
model = UNet(norm_min, norm_max, use_clamp=True).cuda()

# Todo: optimizer settings
decay_epoch = opt.n_epochs / 5  # epoch from which to start lr decay
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.offset, decay_epoch).step
)

L2_Loss = nn.MSELoss()


def get_L2_loss(output, batch_y):
    x = tf_inv(output)
    y = tf_inv(batch_y)
    return L2_Loss(x, y)


def train_val_model():
    loss_min = 1e4
    for epoch in range(opt.n_epochs):
        train_num, test_num = 0, 0
        train_loss_epoch, test_loss_epoch = 0, 0

        model.train()
        for batch_data in tqdm(train_loader):
            images_D = batch_data['D'].cuda()
            images_P = batch_data['P'].cuda()
            # labels = batch_data['label'].cuda()

            output = model(images_D)
            loss = get_L2_loss(output, images_P)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * images_D.shape[0]
            train_num += images_D.shape[0]
        train_loss = train_loss_epoch / train_num
        print('epoch: {}, train loss: {}'.format(epoch + 1, train_loss))

        model.eval()
        for batch_data in tqdm(test_loader):
            images_D = batch_data['D'].cuda()
            images_P = batch_data['P'].cuda()
            # labels = batch_data['label'].cuda()

            output = model(images_D)
            loss = get_L2_loss(output, images_P)

            test_loss_epoch += loss.item() * images_D.shape[0]
            test_num += images_D.shape[0]
        test_loss = test_loss_epoch / test_num
        print('epoch: {}, test loss: {}'.format(epoch + 1, test_loss))

        total_loss = 0.1 * train_loss + 0.9 * test_loss
        if loss_min > total_loss:
            print('Saving...  {}'.format(epoch + 1))
            loss_min = total_loss
            state = {
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, opt.save_path)

        scheduler.step()


train_val_model()

