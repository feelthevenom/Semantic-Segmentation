import os, sys
from segmentation.logging.logger import logging
from segmentation.exception.exception import SegmentationException
from segmentation.constant.config import DATA_DIR, DATA_DIRRS_LIST
from segmentation.constant.config import DATA_ORG_TRAIN_DIR, DATA_ORG_TEST_DIR, DATA_GT_VALID_DIR
from segmentation.constant.config import DATA_GT_TRAIN_DIR, DATA_GT_TEST_DIR, DATA_GT_VALID_DIR

class DatasetConfig:
    def __init__(self):
        pass