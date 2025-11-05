import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils import *

class BCSSDataset(Dataset):
    """
    Custom dataset for the Breast Cancer Semantic Segmentation (BCSS) dataset.
    Corrects the file path issue for the mask images.
    """
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = mask.long()

        return image, mask


if __name__ == '__main__':
    # Create dataset instances
    train_dataset = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=TRANSFORMS_TRAIN)
    val_dataset = BCSSDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform=TRANSFORMS_VAL)

    print(f'Train Sample: {len(train_dataset)}')
    print(f'Validation Sample: {len(val_dataset)}')