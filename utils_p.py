import os
import csv
import shutil
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from my_models import Aug


def get_segment_class_names():
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    print('names: {}'.format(names))
    return names


def image_show(images, title=None):
    if images.shape[0] == 3:
        images = images.detach().cpu().numpy()
        images = images.transpose((1, 2, 0))
        plt.imshow(images)
        if title:
            plt.title(title)
        plt.axis('off')
    elif images.shape[0] == 1:
        images = images.detach().squeeze(0).cpu().numpy()
        plt.imshow(images, cmap='gray')
        if title:
            plt.title(title)
        plt.axis('off')
    plt.show()


def get_eot_images(adv_image, num_samples=32):
    image_out_list = [adv_image]
    aug = Aug()
    for i in range(num_samples):
        adv_image_out = aug(adv_image).clamp(0, 1)
        image_out_list.append(adv_image_out)
    eot_images = torch.stack(image_out_list)
    return eot_images


def get_adv_image_tensor(image, perturb, epsilon):
    perturb_clip = torch.clamp(perturb, -1. * epsilon, epsilon)
    adv_image_tensor = torch.clamp(perturb_clip + image, 0., 1.)
    return adv_image_tensor


def compute_eot_loss_normal(adv_image_tensor, classifier, model_trans, criterion, adv_label,
                            trf_nor, batch_size, aerf=0.6, cmyk_flag=False):
    eot_images = get_eot_images(adv_image_tensor, batch_size)

    loss = torch.tensor(0.).cuda()
    for i in range(eot_images.shape[0]):
        adv_image_D = trf_nor(eot_images[i])
        adv_image_P = model_trans(adv_image_D)

        adv_out_D = classifier(adv_image_D)
        adv_out_P = classifier(adv_image_P)

        if cmyk_flag:
            adv_image_D_cmyk = rgb_to_cmyk_to_rgb(adv_image_D)
            adv_image_P_cmyk = rgb_to_cmyk_to_rgb(adv_image_P)
            adv_out_D_cmyk = classifier(adv_image_D_cmyk)
            adv_out_P_cmyk = classifier(adv_image_P_cmyk)
            loss += (1 - aerf) / 2. * criterion(adv_out_D, adv_label) + aerf / 2. * criterion(adv_out_P, adv_label) + \
                    (1 - aerf) / 2. * criterion(adv_out_D_cmyk, adv_label) + aerf / 2. * criterion(adv_out_P_cmyk, adv_label)
        else:
            loss += criterion(adv_out_D, adv_label) + aerf * criterion(adv_out_P, adv_label)

    loss /= eot_images.shape[0]

    return loss


def rgb_to_cmyk_to_rgb(image):
    image_cmyk = rgb_to_cmyk(image)
    image_rgb = cmyk_to_rgb(image_cmyk)
    return image_rgb


def get_image_predict(classifier, image_tensor, idx2label, num_labels=3):
    classifier.eval()
    out = classifier(image_tensor)
    y_softmax = F.softmax(out, dim=1)
    val, idx = y_softmax[0].sort(descending=True)
    res = [(idx[i].item(), idx2label[idx[i]], round(val[i].item(), 3)) for i in range(num_labels)]
    top1_val, top1_idx = val[0].item(), idx[0].item()
    return top1_idx, top1_val, res


def is_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('path of {} is building.'.format(dir_name))
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print('path of {} already exist and rebuild'.format(dir_name))


def rgb_to_cmyk(image, rgb_scale=1, cmyk_scale=1):
    assert image.shape[0] == 1, 'Only support 1 for now'
    _, _, H, W = image.shape
    image_temp = image.clone().squeeze(0)
    r = image_temp[0, ...]
    g = image_temp[1, ...]
    b = image_temp[2, ...]

    c = torch.zeros(size=(H, W))
    m = torch.zeros(size=(H, W))
    y = torch.zeros(size=(H, W))
    k = torch.ones(size=(H, W)) * cmyk_scale
    if (torch.sum(r) == 0) and (torch.sum(g) == 0) and (torch.sum(b) == 0):
        # black
        return torch.stack([c, m, y, k])

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    temp = torch.stack([c, m, y])
    min_cmy, _ = torch.min(temp, dim=0)
    c = (c - min_cmy)
    m = (m - min_cmy)
    y = (y - min_cmy)
    k = min_cmy

    # rescale to the range [0, cmyk_scale]
    return torch.stack([c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale]).unsqueeze(0)


def cmyk_to_rgb(cmyk, rgb_scale=1, cmyk_scale=1):
    assert cmyk.shape[0] == 1, 'Only support 1 for now'
    _, _, H, W = cmyk.shape
    cmyk_temp = cmyk.clone().squeeze(0)
    c = cmyk_temp[0, ...]
    m = cmyk_temp[1, ...]
    y = cmyk_temp[2, ...]
    k = cmyk_temp[3, ...]

    r = rgb_scale * (1.0 - (c + k) / float(cmyk_scale))
    g = rgb_scale * (1.0 - (m + k) / float(cmyk_scale))
    b = rgb_scale * (1.0 - (y + k) / float(cmyk_scale))
    return torch.stack([r, g, b]).unsqueeze(0)


def get_normal_min_max(mean=None, std=None):
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    min_in, max_in = 0, 1
    min_in, max_in = torch.tensor([min_in, min_in, min_in], dtype=torch.float32), \
                     torch.tensor([max_in, max_in, max_in], dtype=torch.float32)
    mean, std = torch.tensor(mean), torch.tensor(std)

    # -2.1179039301310043, 2.6399999999999997
    norm_min, norm_max = (min_in - mean) / std, (max_in - mean) / std

    return norm_min, norm_max


def show_image_pre(image_path, model, transforms_test, idx2label, device):
    model.eval()
    image = Image.open(image_path)
    # image = image.resize((224, 224), Image.ANTIALIAS)
    image = image.resize((224, 224), Image.BILINEAR)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image_tensor = transforms_test(image)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
    res = get_image_predict(model, image_tensor, idx2label, num_labels=5)
    print('top 5:', res)


def get_index_conf(model, image_tensor, index):
    model.eval()
    out = model(image_tensor)  # [batch_size, number_class]
    percentage = F.softmax(out, dim=1)[0]  # softmax
    return percentage[index].item()

