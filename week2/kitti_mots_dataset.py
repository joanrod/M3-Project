import numpy as np
import os, cv2
import pandas as pd
from tqdm import tqdm
from pycocotools.mask import decode, toBbox, frPyObjects
from detectron2.structures import BoxMode


def kitti_mots_dataset(path, split):
    dataset_dicts = []
    txt = open(split, 'r')
    txt_lines = txt.read().splitlines()

    # Iterate through all the folders which correspond to the dataset
    for line in txt_lines:
        txt_path = os.path.join(path, 'instances_txt', f'{line[-4:]}.txt')  # Path of the sequence folder

        # Open the corresponding instance_txt file
        with open(txt_path) as sequence:
            # Obtain all the data in the file in a pandas.df
            table = pd.read_table(
                filepath_or_buffer=sequence,
                sep=" ",
                header=0,
                names=["image", "instance_id", "class_id", "height", "width", "rle"],
                dtype={"image": int, "instance_id": int, "class_id": int, "height": int, "width": int, "rle": str},
            )

        # Iterate through the images of the corresponding path
        for img_path in os.listdir(os.path.join(path, line)):
            img_num = int(img_path[:-4])    # integer image number
            record = {}                     # Dictionary of the annotations of the current image

            filename = os.path.join(path, line, img_path)
            record["file_name"] = filename                              # Path to the image
            record["image_id"] = f'{line[-4:]}{img_path[:-4]}'          # ImageId as 'SequenceNumberImgNumber' e.g. 0014000245

            image_data = table[table["image"] == img_num]               # Obtain annotations of the current image

            record["height"] = table["height"].iloc[0]                  # image height
            record["width"] = table["width"].iloc[0]                    # image width

            objs = []   # List of dictionaries with the data of all the objects in the current frame

            # Iterate through all the objects of the current image
            for frame_num, obj_id, class_id, h, w, rle in image_data.itertuples(index=False):
                if class_id not in [0, 10]:
                    rle = frPyObjects([bytearray(rle, "utf8")], h, w)
                    # data of the object
                    obj = {
                        "bbox": toBbox(rle).tolist()[0],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": class_id-1,
                        #"segmentation": rle,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

if __name__=="__main__":
    dataset = kitti_mots_dataset('../../data/KITTI-MOTS/', 'kitti_splits/kitti_train.txt')
    print()
