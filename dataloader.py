import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def get_inv_normalize(m, s):
    res = transforms.Normalize(
        mean=[-m[0]/s[0], -m[1]/s[1], -m[2]/s[2]],
        std=[1/s[0], 1/s[1], 1/s[2]],
    )
    return res


class D2PDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.files_D = sorted(glob.glob(os.path.join(root, 'digital') + '/*.*'))
        self.files_P = sorted(glob.glob(os.path.join(root, 'physical') + '/*.*'))
        self.transform = transforms_

    def __getitem__(self, index):
        image_D = Image.open(self.files_D[index])
        image_P = Image.open(self.files_P[index])

        # Get label
        temp = self.files_D[index]
        label = int(temp.split('/')[-1].split('_')[-1].split('.')[0])

        # Convert grayscale images to rgb
        if image_D.mode != "RGB":
            image_D = to_rgb(image_D)
        if image_P.mode != "RGB":
            image_P = to_rgb(image_P)

        image_D = self.transform(image_D)
        image_P = self.transform(image_P)
        label = torch.tensor(label, dtype=torch.long)

        image_path_D = temp
        image_path_P = self.files_P[index]

        return {'D': image_D, 'P': image_P, 'label': label, 'd_path': image_path_D, 'p_path': image_path_P}

    def __len__(self):
        return len(self.files_D)


class im_dataset(Dataset):
    def __init__(self, data_dir, im_size=224):
        self.data_dir = data_dir
        self.imgpaths = self.get_imgpaths()

        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
        ])

    def get_imgpaths(self):
        paths = sorted([os.path.join(self.data_dir, x)
                        for x in os.listdir(self.data_dir) if x.endswith(('JPEG', 'jpg', 'png'))])
        return paths

    def __getitem__(self, idx):
        img_name = self.imgpaths[idx]
        file_name = os.path.splitext(os.path.basename(img_name))[0]
        image = Image.open(img_name)
        image_t = self.transform(image)
        return image_t, file_name

    def __len__(self):
        return len(self.imgpaths)

