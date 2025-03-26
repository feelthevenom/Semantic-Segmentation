from segmentation.utils.dataset.utils import list_datasets

"""
Define contant variable values 
"""

# Dataset Dirr
DATA_DIR = "Dataset"

# Training Dataset Dirr
DATA_ORG_TRAIN_DIR: str = "train" 
DATA_ORG_VALID_DIR: str = "val"
DATA_ORG_TEST_DIR: str = "test"

#Testing Dataset Dirr
DATA_GT_TRAIN_DIR: str= "train_label"
DATA_GT_VALID_DIR: str = "val_label"
DATA_GT_TEST_DIR: str = "test_label"

SELECTED_CLASS = ['background', 'building'] 
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]

IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
N_CLASSES=2 # Output Channel

# Datasets
DATA_DIRRS_LIST = list_datasets()
NUM_OF_DATASET = len(DATA_DIRRS_LIST)