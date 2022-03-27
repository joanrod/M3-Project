import numpy as np
import os, cv2
import pandas as pd
from PIL import Image
import pickle
import glob
from tqdm import tqdm
from pycocotools.mask import decode, toBbox, frPyObjects, encode
from detectron2.structures import BoxMode
import json
import matplotlib.pyplot as plt
from imantics import Polygons, Mask

def saveData(path, data):
    """Save data as pickle file"""
    open_file = open(path, "wb")
    pickle.dump(data, open_file)
    open_file.close()

def loadData(path):
    """Load and read json file"""
    open_file = open(path, "rb")
    data_loaded = pickle.load(open_file)
    open_file.close()
    return data_loaded

def kitti_mots_dataset(path, split, type="inference"):
    """
    Function to load the kitti-mots dataset as coco format
    :param path: string, path to the KITTI-MOTS folder
    :param split: string, path to the corresponding txt which divides the sequence into datasets
    :return: list of annotations
            dataset_dicts = [{'file_name': str
                              'image_id': str
                              'height': int
                              'width': int
                              'annotations': [{'bbox': [x, y, w, h],
                                              'bbox_mode: BoxMode.XYWH_ABS,
                                              'category_id': int,
                                              'segmentation': rle
                                              },
                                              ...
                                              ]
                              ...
                             }]
    """
    # First of all look if the annotations have been already transformed to coco format
    if type == "inference":
        pickle_path = f'{split[:-4]}.pkl'
    else:
        pickle_path = f'{split[:-4]}_finetune.pkl'

    if os.path.exists(pickle_path):
        dataset_dicts = loadData(pickle_path)

    # If it is the first time creating the annotations in coco format, create it
    else:
        dataset_dicts = []                      # List of annotations
        txt = open(split, 'r')                  # Open the text file in which there are the sequences of the corresponding split
        txt_lines = txt.read().splitlines()     # Read all the lines of the text file

        # Iterate through all the folders which correspond to the dataset
        for line in txt_lines:
            folder_path = os.path.join(path, 'instances', f'{line[-4:]}')


            # Iterate through the images of the corresponding path
            for img_path in glob.glob(f'{folder_path}/*.png'):
                record = {}  # Annotations of one image
                mask = np.array(Image.open(img_path))   # mask
                img_num = img_path[-10:-4]              # OJO string not int
                record["file_name"] = os.path.join(path, line, f'{img_num}.png')
                record["image_id"] = f'{line[-4:]}{img_num}'    # The image_id is created by concatenating the sequence number and the image number
                record['height'] = mask.shape[0]
                record['width'] = mask.shape[1]

                objs = []                   # Objects in the current image
                obj_ids = np.unique(mask)   # All the object ids of the mask

                # Iterate through the components of the mask
                for id in obj_ids:
                    if id not in [0, 10000]:    # Skip background and objects to ignore
                        id_mask = np.zeros(mask.shape, dtype=np.uint8)  # mask for the object id
                        id_mask[mask == id] = 1                           # set to 1 only the object id values
                        if type == "inference":
                            class_id = map_kitti_coco(id // 1000)
                        else:
                            class_id = int((id // 1000) - 1)

                        # Object annotations
                        obj = {
                            "bbox": cv2.boundingRect(id_mask),                      # BBox in x y w h
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": class_id,                   # class id (0: car, 1: pedestrian)
                            "segmentation": encode(np.asarray(id_mask, order="F")), # segmentation mask encoded in rle
                        }
                        objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
        saveData(path=pickle_path, data=dataset_dicts)
    return dataset_dicts

def map_kitti_coco(label):
    if label == 2:
        return 0
    elif label == 1:
        return 2
if __name__=="__main__":
    dataset = kitti_mots_dataset('../../data/KITTI-MOTS/', 'kitti_splits/kitti_val.txt')
    print()
