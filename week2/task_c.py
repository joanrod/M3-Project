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

detectron_model = "segmentation" # detection or segmentation

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
if detectron_model == "detection":
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

# Load image from KITTI-MOTS
path_data = '/home/mcv/datasets/KITTI-MOTS/testing/image_02/'

sequence = '0021/'
image_name = '000042'

im = cv2.imread(path_data + sequence+ image_name+'.png')


predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite('task_c_train_0000_'+image_name+'_detections.png', out.get_image()[:, :, ::-1])
cv2.imwrite('task_c_train_0000_'+image_name+'_orig.png', im)
