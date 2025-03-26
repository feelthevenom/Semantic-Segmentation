import os,sys
import torch 


from segmentation.models.msunet.model import MSU_Net
from segmentation.models.bmsunet.model import BMSU_Net

from segmentation.logging.logger import logging
from segmentation.exception.exception import SegmentationException

from segmentation.entity.config_entity import DatasetConfig
from segmentation.utils.dataset.utils import data_loader, training_setup
from segmentation.utils.main_utils.utils import save_metrics, plot_metrics

from segmentation.constant.config import EPOCHS, LR_RATE, MODEL_LOSS, MODEL_METRICS
from segmentation.constant.config import ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, BEST_MODEL_NAME, TRAIN_METRIC_DIRR



logger = logging.getLogger('Model_Training')

class DataTransformingConfig:
    def __init__(self):
        self.dataset_config = DatasetConfig()

class ModelTraining:
    def __init__(self):
        self.models = {
            "MSU_Net": MSU_Net(),
            "BMSU_Net": BMSU_Net()
        }
        self.config = DataTransformingConfig()

    def train_function(self, model, train_loader, valid_loader, model_name):
        try:
            logger.info("Entering Training Processes")
            # Set device: `cuda` or `cpu`
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Define loss function
            loss = MODEL_LOSS

            # Define metrics
            metrics = MODEL_METRICS 

            logger.info("Entering Model Training Setup")
            # Setup training
            train_epoch, valid_epoch = training_setup(
                model=model, lrate=LR_RATE, loss=loss, metrics=metrics, DEVICE=DEVICE
            )
            logger.info("Model training setup is successful!")

            # Define paths
            path = os.path.join(ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, model_name)
            model_path = os.path.join(path, BEST_MODEL_NAME)
            metrics_file_path = os.path.join(path, TRAIN_METRIC_DIRR)
            os.makedirs(path, exist_ok=True)
            save_dir = os.path.join(ARTIFACT_DIRR_NAME, TRAINED_MODEL_DIRR, model_name, "plots")
            # Training loop
            TRAINING = True
            if TRAINING:
                modelNum = 0  # Stores the best model's epoch number.
                best_iou_score = 0.0
                logger.info("Model Training is Started")

                for i in range(EPOCHS):
                    print(f'\nEpoch: {i + 1}')

                    # Perform training & validation
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)

                    # Store metrics for each epoch
                    metrics_data = {
                        "epoch": i + 1,
                        "train_metrics": train_logs,
                        "valid_metrics": valid_logs
                    }

                    # Save metrics for every epoch
                    save_metrics(metrics_data, metrics_file_path)

                    # Save model if a better val IoU score is obtained
                    if best_iou_score < valid_logs['iou_score']:
                        modelNum = i
                        best_iou_score = valid_logs['iou_score']

                        torch.save(model, model_path)  # Save the entire model
                        print('Model saved!')

                logger.info('Best Model is Saved')
                # Generate and save plots
                plot_metrics(metrics_file_path, save_dir)
                logger.info("Model Plots are saved")
        except Exception as e:
            raise SegmentationException(e, sys)


    def train_all_datasets(self):
        """
        Train all datasets sequentially for each model.

        Args:
        - train_function: Function that takes (model, train_loader, valid_loader) as arguments.
        """
        try:

            dataset_config = self.config.dataset_config

            for dataset_idx, (x_train, y_train, x_valid, y_valid) in enumerate(
                zip(dataset_config.org_train_dirr, dataset_config.gt_train_dirr,
                    dataset_config.org_valid_dirr, dataset_config.gt_valid_dirr)
            ):
                logger.info(f"Training on Dataset {dataset_idx + 1}/{len(dataset_config.all_dataset)}: {x_train}")

                # Load dataset
                try:
                    train_loader, valid_loader = data_loader(
                        x_train_dir=x_train, y_train_dir=y_train,
                        x_valid_dir=x_valid, y_valid_dir=y_valid
                    )
                except Exception as e:
                    logger.error(f"Failed to load dataset {dataset_idx + 1}: {str(e)}")
                    continue  # Skip this dataset and move to the next one  

                # Train each model sequentially on the current dataset
                for model_name, model in self.models.items():
                    try:
                        logger.info(f"Training {model_name} on Dataset {dataset_idx + 1}")

                        # Train model
                        self.train_function(model, train_loader, valid_loader, model_name=model_name)

                        logger.info(f"Finished training {model_name} on Dataset {dataset_idx + 1}")

                    except Exception as e:
                        logger.error(f"Error training {model_name} on dataset {dataset_idx + 1}: {str(e)}")
                        raise SegmentationException(e, sys)
                    
        except Exception as e:
            raise SegmentationException(e, sys)
        
