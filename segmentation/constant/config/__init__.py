
"""
Define contant variable values 
"""

# Dataset Directories
DATA_DIR = "Dataset"

# Training Dataset Directories
DATA_ORG_TRAIN_DIR = "train" 
DATA_ORG_VALID_DIR = "val"
DATA_ORG_TEST_DIR = "test"

# Testing Dataset Directories
DATA_GT_TRAIN_DIR = "train_label"
DATA_GT_VALID_DIR = "val_label"
DATA_GT_TEST_DIR = "test_label"

SELECTED_CLASS = ['background', 'building']
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]

IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
OUT_CLASSES = 2  # Output Channels

# Preprocessed Dataset Directory
PREPROCESS_OUT_DIRR = "Preprocessed_Data"

# Artifact Folder Directory
ARTIFACT_DIRR_NAME = "Artifact"
TRAINED_MODEL_DIRR = "Trained Models"
BEST_MODEL_NAME = "model.pth"

PRED_IMGS_DIRR = "Predicted"

#PARAMS
BATCH_NUM = 8
EPOCHS = 1
LR_RATE = 0.001