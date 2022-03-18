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
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from kitti_mots_dataset import kitti_mots_dataset

kitti_path = '../../data/KITTI-MOTS/'
kitti_correspondences = {
    'Car': 1,
    'Pedestrian': 2,
}

# Register KITTI dataset
for d in ["train", "val", "test"]:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt"))
    MetadataCatalog.get("KITTI-MOT_" + d).set(thing_classes=["car", "pedestrian"])
# TO DO: Try different ones to benhcmark (3 for Faster and 3 for Mask)
model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #This models gives best results based on KITTI-MOTS tables

# Posible ayuda
# https://github.com/kevinbtw-codes/KITTI-Multi-Object-Detection-Tracking-Detectron2/blob/main/KITTI-MOTS-Detectron2.ipynb

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_id))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
cfg.DATASETS.TEST = ("KITTI-MOTS_test",)
predictor = DefaultPredictor(cfg)

dataset_dicts = kitti_mots_dataset(path=kitti_path, split="kitti_splits/kitti_train.txt")

print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
for d in random.sample(dataset_dicts, 4):
    im_path = d["file_name"]
    img = cv2.imread(im_path)

    outputs = predictor(img)

    v = Visualizer(img[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
    out = out.get_image()[:,:,::-1]
    cv2.imwrite('test.jpg', out)

evaluator = COCOEvaluator("KITTI-MOTS_val", output_dir='inference/')
val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

# cfg = get_cfg()
# cfg.merge_from_file((model_zoo.get_config_file(model_id)))
# cfg.DATASETS.TRAIN = ("KITTI-MOTS_train")
# cfg.DATASETS.TEST = ("KITTI-MOTS_val")
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()