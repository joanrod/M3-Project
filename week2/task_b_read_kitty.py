import os

import PIL.Image as Image
import numpy as np

path_data = '../../data/KITTI-MOTS/instances/0000/'


for filename in os.listdir(path_data):
    if filename.endswith('png'):
        print(filename)
        img = np.array(Image.open(path_data + filename))
        obj_ids = np.unique(img)
        # to correctly interpret the id of a single object
        obj_id = obj_ids[1]
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000
        print(img.shape)
        print( obj_ids, obj_id, class_id, obj_instance_id)