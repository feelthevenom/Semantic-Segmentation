import os, sys
from segmentation.logging.logger import logging
from segmentation.exception.exception import SegmentationException

from segmentation.constant.config import DATA_DIR
from segmentation.utils.dataset.utils import DATA_DIRRS_LIST
from segmentation.constant.config import DATA_ORG_TRAIN_DIR, DATA_ORG_TEST_DIR, DATA_ORG_VALID_DIR
from segmentation.constant.config import DATA_GT_TRAIN_DIR, DATA_GT_TEST_DIR, DATA_GT_VALID_DIR


class DatasetConfig:
    def __init__(self):
        self.all_dataset = DATA_DIRRS_LIST
        self.org_train_dirr = [os.path.join(DATA_DIR, train_dirr, DATA_ORG_TRAIN_DIR) for train_dirr in self.all_dataset]
        self.org_valid_dirr = [os.path.join(DATA_DIR, valid_dirr, DATA_ORG_VALID_DIR) for valid_dirr in self.all_dataset]
        self.org_test_dirr = [os.path.join(DATA_DIR, test_dirr, DATA_ORG_TEST_DIR) for test_dirr in self.all_dataset]
        self.gt_train_dirr = [os.path.join(DATA_DIR, train_dirr, DATA_GT_TRAIN_DIR) for train_dirr in self.all_dataset]
        self.gt_valid_dirr = [os.path.join(DATA_DIR, valid_dirr, DATA_GT_VALID_DIR) for valid_dirr in self.all_dataset]
        self.gt_test_dirr = [os.path.join(DATA_DIR, test_dirr, DATA_GT_TEST_DIR) for test_dirr in self.all_dataset]
