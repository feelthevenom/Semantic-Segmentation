import os,sys
import torch 
import numpy as np
import cv2

from segmentation.models.msunet.model import MSU_Net
from segmentation.models.bmsunet.model import BMSU_Net

from segmentation.logging.logger import logging
from segmentation.exception.exception import SegmentationException

from segmentation.utils.dataset.utils import BuildingsDataset
from segmentation.entity.config_entity import DatasetConfig

from segmentation.utils.augmentation.utils import get_preprocessing, get_validation_augmentation

from segmentation.constant.config import CLASS_RGB_VALUES
from segmentation.utils.augmentation.utils import crop_image
from segmentation.utils.helper.utils import colour_code_segmentation, reverse_one_hot

from segmentation.constant.config import ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, BEST_MODEL_NAME, PRED_IMGS_DIRR

from torch.utils.data import DataLoader

logger = logging.getLogger('Prediction_Pipeline')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelPrediction:
    def __init__(self):
        self.config = DatasetConfig()
        self.models = {
            "MSU_Net": MSU_Net(),
            "BMSU_Net": BMSU_Net()
        }

    
    def load_training(self, x_test_dir, y_test_dir):
        try:
            test_dataset = BuildingsDataset(
                x_test_dir,
                y_test_dir,
                augmentation=get_validation_augmentation(),
                preprocessing=get_preprocessing(preprocessing_fn=None),
                class_rgb_values=CLASS_RGB_VALUES
            )
            test_dataloader = DataLoader(test_dataset)

            # Test dataset for visualization (without preprocessing transformations)
            test_dataset_vis = BuildingsDataset(
                x_test_dir, y_test_dir,
                augmentation=get_validation_augmentation(),
                class_rgb_values=CLASS_RGB_VALUES
            )
            return test_dataset, test_dataloader, test_dataset_vis
        except Exception as e:
            raise SegmentationException(e, sys)
        
        # best_model = torch.load(model_path, map_location=DEVICE)
    def predict(self):
        try:
            for dataset_idx, (x_test_dir, y_test_dir) in enumerate(
                zip(self.config.org_test_dirr, self.config.gt_test_dirr)
            ):
                test_dataset, test_dataloader, test_dataset_vis = self.load_training(x_test_dir, y_test_dir)
                for model_name, model in self.models.items():
                    logger.info(f"Prediction for {model_name}")
                    print(f"Prediction for {model_name}")
                    sample_preds_folder = os.path.join(ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, model_name, PRED_IMGS_DIRR)
                    os.makedirs(sample_preds_folder, exist_ok=True)
                    model_path = os.path.join(ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, model_name, BEST_MODEL_NAME)
                    
                    # Contains the list Model Classes
                    model_classes = list({type(model) for model in self.models.values()})

                    # Import MaxPool2d and register safe globals
                    from torch.nn.modules.pooling import MaxPool2d
                    globals_to_allow = model_classes + [MaxPool2d]

                    # Register the safe globals
                    torch.serialization.add_safe_globals(globals_to_allow)

                    # Load checkpoint with weights_only set to False
                    best_model = torch.load(model_path, map_location=DEVICE, weights_only=False)
                    
                    with torch.no_grad():
                        for idx in range(len(test_dataset)):
                            image, gt_mask = test_dataset[idx]
                            image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
                            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

                            # Predict test image
                            pred_mask = best_model(x_tensor)
                            pred_mask = pred_mask.detach().squeeze().cpu().numpy()

                            # Convert pred_mask from CHW to HWC format
                            pred_mask = np.transpose(pred_mask, [1, 2, 0])

                            # Get prediction channel corresponding to building
                            pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES))

                            # Convert gt_mask from CHW to HWC format
                            gt_mask = np.transpose(gt_mask, [1, 2, 0])
                            gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), CLASS_RGB_VALUES))

                            cv2.imwrite(os.path.join(sample_preds_folder, f"pred_{idx}.png"), pred_mask)
            
        except Exception as e:
            raise SegmentationException(e, sys)

