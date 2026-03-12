import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class FIVESDataset(TorchDataset):
    def __init__(self,
                 img_ids,
                 img_dir,
                 mask_dir=None,
                 img_ext='.png',
                 mask_ext='.png',
                 transform=None,
                 return_mask=True):
        """
        img_ids: list of image names WITHOUT extension
        img_dir: path to image folder
        mask_dir: path to mask folder
        return_mask: False for test/inference
        """

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.return_mask = return_mask

        # Classification label mapping
        self.label_map = {
            "N": 0,   # Normal
            "D": 1,   # Diabetic Retinopathy
            "G": 2,   # Glaucoma
            "A": 3    # AMD (Age-related Macular Degeneration)
        }

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        # ---------------------------------
        # CLEAN filename (remove " (1)")
        # ---------------------------------
        clean_img_id = img_id.split(" (")[0]

        # ---------------------------------
        # Load Image
        # ---------------------------------
        img_path = os.path.join(self.img_dir, clean_img_id + self.img_ext)

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------------------------------
        # Extract Classification Label
        # ---------------------------------
        raw_label = clean_img_id.split("_")[-1]

        # Safety check
        if raw_label not in self.label_map:
            raise ValueError(f"Unknown label extracted: {raw_label} from {img_id}")

        label = torch.tensor(self.label_map[raw_label]).long()

        # ---------------------------------
        # Load Mask (if required)
        # ---------------------------------
        if self.return_mask and self.mask_dir is not None:

            mask_path = os.path.join(self.mask_dir, clean_img_id + self.mask_ext)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found: {mask_path}")

            mask = mask.astype('float32') / 255.0
            mask = np.expand_dims(mask, axis=-1)

            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            img = img.astype('float32') / 255.0
            img = img.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)

            return torch.tensor(img).float(), torch.tensor(mask).float(), label

        else:
            # For test/inference (no mask)
            if self.transform is not None:
                augmented = self.transform(image=img)
                img = augmented['image']

            img = img.astype('float32') / 255.0
            img = img.transpose(2, 0, 1)

            return torch.tensor(img).float(), label
