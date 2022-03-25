"""
Apply pre-trained MaskRCNN or FasterRCNN on COCO transplanting new objects by co-occurrence
"""
import os
import glob
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

model_id = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
taskb_path = 'taskb_input/'

if __name__ == "__main__":

    # CONFIGURATION
    # Model config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id))    # model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)  # Model

    # Hyper-params
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)   # Initialize predictor

    os.makedirs('taskb_output', exist_ok=True)
    # Iterate through all the images of the dataset
    for idx, img_path in enumerate(sorted(glob.glob(f'{taskb_path}/*.jpg'))):
        im = cv2.imread(img_path)

        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

        out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
        cv2.imwrite(f'taskb_output/{idx}.png', out.get_image()[:, :, ::-1])
