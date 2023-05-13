import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset:
    def __init__(self, images_path, labels, resize=None):
        # resize --> (h, w)
        self.images_path = images_path
        self.labels = labels
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image = Image.open(self.images_path[item]).convert("RGB")
        label = self.labels[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = self.aug(image=np.array(image))['image']

        # make channels first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {'images': torch.tensor(image, dtype=torch.float),
                'labels': torch.tensor(label, dtype=torch.long)}
