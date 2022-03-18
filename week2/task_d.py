import matplotlib.pyplot as plt
import torch
# Some basic setup:
# Setup detectron2 logger
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

from kitti_mots_dataset import kitti_mots_dataset

kitti_path = '../../data/KITTI-MOTS/'

# Register KITTI dataset
for d in ["train", "val", "test"]:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt"))
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=["car, pedestrian"])

# TO DO: Try different ones to benhcmark (3 for Faster and 3 for Mask)
model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #This models gives best results based on KITTI-MOTS tables

# Obtain

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_id))
cfg.DATASETS.TRAIN = ("kitti_train",)
cfg.DATASETS.VAL = ("kitti_val",)
cfg.DATASETS.TEST = ("kitti_test",)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2