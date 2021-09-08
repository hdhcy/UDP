import kornia

import random
import torch.nn as nn

"""
EOT-S operation
"""


class Aug(nn.Module):
    def __init__(self):
        super(Aug, self).__init__()

        self.eot_transform = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            # kornia.augmentation.RandomVerticalFlip(p=0.5),
            kornia.augmentation.ColorJitter(
                hue=0.1, saturation=0.2, brightness=0.3, contrast=0.5, p=0.85
            ),
            kornia.augmentation.RandomPerspective(distortion_scale=0.15, p=0.85),
            kornia.augmentation.RandomAffine(
                degrees=[-30, 30], translate=0.1, scale=[0.7, 0.9], shear=[-5, 5, -5, 5], p=0.85
            ),
            kornia.augmentation.RandomMotionBlur(
                kernel_size=(3, 7), angle=[-15, 15], direction=[-0.5, 0.5], p=0.85,
            ),
        )
        self.resize_224_224 = kornia.Resize(size=(224, 224), align_corners=False)

    def forward(self, image):
        # image = image.cpu()
        random_resize = int(random.uniform(4, 10) * 224)
        rand_size_x = random_resize + int(random.uniform(-0.1, 0.1) * random_resize)
        rand_size_y = random_resize + int(random.uniform(-0.1, 0.1) * random_resize)
        resize = kornia.Resize(size=(rand_size_x, rand_size_y), align_corners=False)

        gau_std = random.uniform(0, 0.02)
        gaussion = kornia.augmentation.RandomGaussianNoise(std=gau_std, p=0.5)
        kernel_size_list = [(3, 3), (5, 5), (7, 7)]
        gaussion_blur_kernel_size = kernel_size_list[random.randint(0, 2)]
        gaussion_blur_sigma = (random.randint(1, 3) / 255., random.randint(1, 3) / 255.)
        gaussion_blur = kornia.augmentation.GaussianBlur(kernel_size=gaussion_blur_kernel_size,
                                                         sigma=gaussion_blur_sigma)

        out = self.eot_transform(image)
        out = gaussion_blur(out)
        out = resize(out)
        out = gaussion(out)
        out = self.resize_224_224(out)

        return out

