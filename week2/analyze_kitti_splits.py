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

kitti_path = '../../data/KITTI-MOTS/'
results_path = 'results/task_d/'

# Register KITTI dataset (train, val and test) with corresponding classes
for d in ["train", "val", "test"]:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: kitti_mots_dataset(kitti_path, "kitti_splits/kitti_" + d + ".txt"))
    MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["person", "car"])
tot_cars = 0
tot_ped = 0
count_masks_per_sequence = {"0001": {"car":0, "person":0},
                            "0002":{"car":0, "person":0},
                            "0003":{"car":0, "person":0},
                            "0004":{"car":0, "person":0},
                            "0005":{"car":0, "person":0},
                            "0006":{"car":0, "person":0},
                            "0007":{"car":0, "person":0},
                            "0008":{"car":0, "person":0},
                            "0009":{"car":0, "person":0},
                            "0010":{"car":0, "person":0},
                            "0011":{"car":0, "person":0},
                            "0012":{"car":0, "person":0},
                            "0013":{"car":0, "person":0},
                            "0014":{"car":0, "person":0},
                            "0015":{"car":0, "person":0},
                            "0016":{"car":0, "person":0},
                            "0017":{"car":0, "person":0},
                            "0018":{"car":0, "person":0},
                            "0019":{"car":0, "person":0},
                            "0020":{"car":0, "person":0},
                            "0000":{"car":0, "person":0}}
for split in ["train", "val", "test"]:
    dataset = DatasetCatalog.get("KITTI-MOTS_" + split)
    count_categories = {"car":0, "person":0}
    for item in dataset:
        image_seq = item.get("image_id")[:4]

        for anot in item.get("annotations"):
            category = anot["category_id"]
            if category == 0:
                count_categories["person"] += 1
                count_masks_per_sequence[str(image_seq)]["person"] += 1
                tot_ped += 1
            else:
                count_categories["car"] += 1
                count_masks_per_sequence[str(image_seq)]["car"] += 1
                tot_cars += 1

    print(split, ": ", count_categories)
    print("tot cars: ", tot_cars, " " + "tot oed: ", tot_ped, "prop: ", tot_ped/tot_cars)

    print(count_masks_per_sequence)