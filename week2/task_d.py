from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2, random
import copy
import torch
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer

from kitti_mots_dataset import kitti_mots_dataset
from utils import *

kitti_path = '../../data/KITTI-MOTS/'
kitti_correspondences = {
    'Car': 1,
    'Pedestrian': 2,
}

# Register KITTI dataset (train, val and test) with corresponding classes
for d in ["train", "val", "test"]:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt"))
    MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["car", "pedestrian"])

# TO DO: Try different ones to benhcmark (3 for Faster and 3 for Mask)
model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #This models gives best results based on KITTI-MOTS tables


# CONFIGURATION
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_id))    # model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Model

cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
cfg.DATASETS.TEST = ("KITTI-MOTS_test",)
cfg.INPUT.MASK_FORMAT = 'bitmask'                           # segmentation as encoded binary mask (rle)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0002 * 2 * 1.4 / 16 # learning rate
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # batch size per image
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # threshold used to filter out low-scored bounding boxes in predictions
cfg.MODEL.DEVICE = "cuda"
cfg.OUTPUT_DIR = 'output'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])
print(trainer._hooks)
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
print(trainer._hooks)

trainer.resume_or_load(resume=False)
trainer.train()

torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodel.pth"))

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "mymodel.pth")
trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])

trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

# If resume=True, it will load the previous trained model. If this time the number of epochs is higher it will
# start trianing from the loaded model. If the number of epochs is lower than the previous model, the model will not
# be trained. If the model do not exists, the training will start from scratch
trainer.resume_or_load(resume=False)

checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)

predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(trainer.model.state_dict())

dataset_dicts = kitti_mots_dataset(path=kitti_path, split="kitti_splits/kitti_test.txt")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

evaluator = COCOEvaluator("KITTI-MOTS_test", cfg, False, output_dir='output')
val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
#show_results(cfg, dataset_dicts, predictor, samples=10)

# predictor = DefaultPredictor(cfg)                           # Create predictor
#
# # Evaluate (INFERENCE)
# evaluator = COCOEvaluator("KITTI-MOTS_train", output_dir='inference/')
# val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_train")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))

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