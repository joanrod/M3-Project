import numpy as np
import os, cv2
import pandas as pd
from PIL import Image
import glob
from tqdm import tqdm
from pycocotools.mask import decode, toBbox, frPyObjects
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt


def kitti_mots_dataset(path, split):
    dataset_dicts = []
    txt = open(split, 'r')
    txt_lines = txt.read().splitlines()

    # Iterate through all the folders which correspond to the dataset
    for line in txt_lines:
        folder_path = os.path.join(path, 'instances', f'{line[-4:]}')

        record = {}     # Annotations of one image

        # Iterate through the images of the corresponding path
        for img_path in glob.glob(f'{folder_path}/*.png'):
            mask = np.array(Image.open(img_path))   # mask
            img_num = img_path[-10:-4]              # OJO string not int

            record["file_name"] = os.path.join(path, line, f'{img_num}.png')
            record["image_id"] = f'{line[-4:]}{img_num}'    # The image_id is created by concatenating the sequence number and the image number
            record['height'] = mask.shape[0]
            record['width'] = mask.shape[1]

            objs = []   # Objects in the current image
            obj_ids = np.unique(mask)
            print(obj_ids)
            # Iterate through the components of the mask
            for id in obj_ids:
                if id not in [0, 10000]:
                    id_mask = np.zeros(mask.shape, dtype=np.uint8)
                    id_mask[mask==id] = 1

                    plt.imshow(id_mask)
                    plt.show()

                    obj = {

                    }





    return dataset_dicts

if __name__=="__main__":
    dataset = kitti_mots_dataset('../../data/KITTI-MOTS/', 'kitti_splits/kitti_train.txt')
    print()
