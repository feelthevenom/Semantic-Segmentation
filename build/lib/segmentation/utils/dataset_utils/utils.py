import os, cv2
import torch
import sys
from typing import List

import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from segmentation.exception.exception import SegmentationException
from segmentation.logging.logger import get_logger

from segmentation.utils.helper.utils import one_hot_encode

from segmentation.utils.augmentation.utils import get_preprocessing, get_training_augmentation, get_validation_augmentation
from segmentation.constant.config import CLASS_RGB_VALUES, BATCH_NUM


logger = get_logger('DATASET_UTILS')

def list_datasets()-> List:
    try:
        return list(os.listdir("Dataset"))

    except Exception as e:
        raise SegmentationException(e, sys)

DATA_DIRRS_LIST = list_datasets()


class BuildingsDataset(torch.utils.data.Dataset):
    """
    Massachusetts Buildings Dataset. Reads images, applies augmentation and preprocessing transformations.

    Arguments:
        images_dir (str) : path to images folder
        masks_dir (str) : path to segmentation masks folder
        class_rgb_values (list) : RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose) : data transformation pipeline (e.g., flip, scale, etc.)
        preprocessing (albumentations.Compose) : data preprocessing (e.g., normalization, shape manipulation, etc.)
    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(('.tiff', '.png', '.jpg', '.jpeg'))}
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if f.lower().endswith(('.tiff', '.png', '.jpg', '.jpeg'))}

        # Find common filenames that exist in both folders
        common_filenames = sorted(set(image_files.keys()) & set(mask_files.keys()))

        # Create full paths ensuring correct matching
        self.image_paths = [os.path.join(images_dir, image_files[name]) for name in common_filenames]
        self.mask_paths = [os.path.join(masks_dir, mask_files[name]) for name in common_filenames]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        try:
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

            mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask
        except Exception as e:
            raise SegmentationException(e, sys)
    def __len__(self):
        return len(self.image_paths)


# Use this load the preprocessed dataset

''' function to create train and validation dataloader. '''
# BATCH SIZE 16
def data_loader(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, preprocess_fn=None,b_size=8):
    try:
        # Get train and val dataset instances.
        train_dataset = BuildingsDataset(
                                        x_train_dir, y_train_dir,
                                        augmentation=get_training_augmentation(),
                                        preprocessing=get_preprocessing(preprocess_fn),
                                        class_rgb_values=CLASS_RGB_VALUES
                                        )

        valid_dataset = BuildingsDataset(
                                        x_valid_dir, y_valid_dir,
                                        augmentation=get_validation_augmentation(),
                                        preprocessing=get_preprocessing(preprocess_fn),
                                        class_rgb_values=CLASS_RGB_VALUES
                                        )

        # Get train and val data loaders.
        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        return train_loader, valid_loader
    except Exception as e:
        raise SegmentationException(e, sys)
''' function to set up train & valid epochs. '''

def training_setup(model, lrate, loss, metrics, DEVICE):
    try:
        # define optimizer.
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lrate)])

        # define learning rate scheduler (not used in this NB).
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True
        )
        return train_epoch, valid_epoch
    except Exception as e:
        raise SegmentationException(e, sys)
