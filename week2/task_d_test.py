from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
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

kitti_path = '/export/home/mcv/datasets/KITTI-MOTS/'
results_path = 'results/task_d/'

# Evaluate pretrained in COCO

# Pretrained models
#model_id = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" #(fast)
#model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #(accurate)
#model_id = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" #(fast)
model_id = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" #(accurate)


cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(model_id))  # model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    model_id)  # Model
cfg.SOLVER.IMS_PER_BATCH = 2
predictor = DefaultPredictor(cfg)  # Create predictor

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = metadata.thing_classes

for d in ["train", "val", "test"]:
    DatasetCatalog.register("KITTI-MOTS_" + d,
                            lambda d=d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt"))
    MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=thing_classes)
cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
cfg.DATASETS.TEST = ("KITTI-MOTS_test",)
cfg.INPUT.MASK_FORMAT = 'bitmask'  # segmentation as encoded binary mask (rle)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = len(thing_classes)

cfg.OUTPUT_DIR = results_path + model_id.split('/')[1].split('.')[0]
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

evaluator = COCOEvaluator("KITTI-MOTS_test", output_dir=cfg.OUTPUT_DIR)
test_loader = build_detection_test_loader(cfg, "KITTI-MOTS_test")

print(inference_on_dataset(predictor.model, test_loader, evaluator))

