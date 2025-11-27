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
        
        # # Mapping for BCSS classes
        # # Original values: 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19, 20
        # # Missing: 8, 12, 16, 17
        # self.mapping = np.zeros(21, dtype=np.int64)
        # for i in range(8): self.mapping[i] = i
        # self.mapping[9] = 8
        # self.mapping[10] = 9
        # self.mapping[11] = 10
        # self.mapping[13] = 11
        # self.mapping[14] = 12
        # self.mapping[15] = 13
        # self.mapping[18] = 14
        # self.mapping[19] = 15
        # self.mapping[20] = 16

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # # Apply mapping
        # mask = self.mapping[mask]

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