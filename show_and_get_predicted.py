import json
import torch
from torchvision import models
import torchvision.transforms as transforms

import utils_p


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Todo: Image transformations
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


# TODO: loading classifier model
print('==>Loading model...')
model = models.resnet50(pretrained=True).to(device)
# model = models.inception_v3(pretrained=True).to(device)
model.eval()

image_path = 'physical_adv_images/adv_0157_931.png'
utils_p.show_image_pre(image_path, model, transforms_test, idx2label, device)

