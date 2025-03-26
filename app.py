import streamlit as st
import os, sys
import json

from segmentation.logging.logger import logging
from segmentation.exception.exception import SegmentationException

from segmentation.utils.ui.utils import dataset_chooser
    
if __name__=='__main__':

    choosen_dataset = dataset_chooser()
