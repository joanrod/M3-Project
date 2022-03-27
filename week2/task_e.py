from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2, random
#import wandb
import glob
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

TRAIN = True

#model_id = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" #(fast)
#model_id = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #(accurate)
#model_id = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" #(fast)
model_id = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" #(accurate)


if __name__ == "__main__":
    kitti_path = '/export/home/mcv/datasets/KITTI-MOTS/'
    kitti_correspondences = {
        'Car': 1,
        'Pedestrian': 2,
    }

    #if TRAIN:
        # Init wandb
        #wandb.init(project="detectron2-week2", entity='celulaeucariota', name=model_id, sync_tensorboard=True)

    # Register KITTI dataset (train, val and test) with corresponding classes
    # The datasets are not created here, they are just parsed to the DatsetCatalog, whenever they are used, the function
    # kitti_mots_dataset is called. It will return a list with the annotations
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("KITTI-MOTS_" + d, lambda d = d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt", "finetune"))
        MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["car", "pedestrian"])

    # CONFIGURATION
        # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id))    # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Model

        # Dataset cofig
    cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
    cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
    cfg.DATASETS.TEST = ("KITTI-MOTS_test",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.INPUT.MASK_FORMAT = 'bitmask'                           # segmentation as encoded binary mask (rle)

        # Hyper param config
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001 # learning rate
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # batch size per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if TRAIN:
        # Trainer object in charge of training
        # trainer = MyTrainer(cfg)
        trainer = DefaultTrainer(cfg)

        # Hooks allow you to flexibly decide what the model does during training.
        # The object trainer has a protected attribute called _hooks which contains a list with all the hooks. By default
        # it contains 5 elements:
        #   - IterationTimer object
        #   - LRScheduler object
        #   - PeriodicCheckpointer object
        #   - EvalHook object
        #   - PeriodicWriter
        # So, we add the ValidationLoss Hook to check the validation loss after every iteration. If the loss is lower than
        # the previous best loss, save the model of the current iteration
        val_loss = ValidationLoss(cfg)
        trainer.register_hooks([val_loss])
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1] # Change the position of the ._hooks

        # If resume=True, it will load the previous trained model. If this time the number of epochs is higher it will
        # start trianing from the loaded model. If the number of epochs is lower than the previous model, the model will not
        # be trained. If the model do not exists, the training will start from scratch
        trainer.resume_or_load(resume=True)
        trainer.train()

        # Save the best model in the cfg.OUTPUT_DIR ('output')
        torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"{model_id[:-4].replace('/','-')}.pth"))

    # ONCE the model has been trained:
    # --- INFERENCE ---
    # Threshold of confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"{model_id[:-4].replace('/','-')}.pth")
    predictor = DefaultPredictor(cfg)

    dataset_dicts = kitti_mots_dataset(path=kitti_path, split="kitti_splits/kitti_test.txt")
    kitti_metadata = MetadataCatalog.get('KITTI-MOTS_test')

    #show_results(dataset_dicts, kitti_metadata, predictor, samples=5)

    #predictions_to_video(kitti_path, "kitti_splits/kitti_test.txt", 6, cfg, predictor)

    # Evaluate the TEST set with the COCO metrics
    evaluator = COCOEvaluator("KITTI-MOTS_test", cfg, False, output_dir='output')
    val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))