# prediction.py (Completed)
import os, sys
import torch
import numpy as np
import cv2

from segmentation.models.msunet.model import MSU_Net
from segmentation.models.bmsunet.model import BMSU_Net
from segmentation.logging.logger import get_logger
from segmentation.exception.exception import SegmentationException
from segmentation.utils.augmentation.utils import get_preprocessing, crop_image
from segmentation.utils.helper.utils import colour_code_segmentation, reverse_one_hot
from segmentation.constant.config import ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, BEST_MODEL_NAME, CLASS_RGB_VALUES

logger = get_logger('Prediction')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Prediction:
    def __init__(self, model_name="MSU_Net"):
        try:
            if model_name == "MSU_Net":
                self.model = MSU_Net()
            elif model_name == "BMSU_Net":
                self.model = BMSU_Net()
            else:
                raise ValueError("Unsupported model")

            model_path = os.path.join(ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, model_name, BEST_MODEL_NAME)

            # Allow custom classes for loading
            from torch.nn.modules.pooling import MaxPool2d
            torch.serialization.add_safe_globals([type(self.model), MaxPool2d])

            self.model = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.model.to(DEVICE)
            self.model.eval()

        except Exception as e:
            raise SegmentationException(e, sys)

    def predict_image(self, image_path):
        try:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            original_image = image.copy()

            # Preprocessing
            preprocessing_fn = get_preprocessing(preprocessing_fn=None)
            sample = preprocessing_fn(image=image)
            image = sample['image']

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                pred_mask = self.model(x_tensor)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                pred_mask = np.transpose(pred_mask, [1, 2, 0])
                pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES))

            return original_image, pred_mask

        except Exception as e:
            raise SegmentationException(e, sys)
