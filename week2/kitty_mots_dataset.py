import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def kitti_mots_dataset(split):

    BASE_PATH = '/home/mcv/datasets/KITTI-MOTS/' # KITTI-MOTS dataset location
    SPLIT_FILE = open('kitti_splits/kitti_'+split+'.txt', 'r') # Read file containing sequences of this split
    GT_PATH = BASE_PATH +
    dataset_samples = []
    for item in SPLIT_FILE.readlines():
        dataset_samples.append(item)

    for sample in dataset_samples:
        for image_file in os.listdir(BASE_PATH + sample):
            print(image_file)
            record = {}
            height, width = cv2.imread(image_file).shape[:2]

            record['file_name'] = image_file
            record['image_id'] = image_file # This should be an ID composed by sequence and frame
            record['height'] = height
            record['width'] = width


