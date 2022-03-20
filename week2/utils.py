import torch
import copy
import random
import cv2

from detectron2.engine import HookBase
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.utils import comm   # For multi-gpu communication
from detectron2.utils.visualizer import Visualizer

class ValidationLoss(HookBase):
    """
    Hooks allow you to flexibly decide what the model does during triaining (e.g. after each step, before each step
    etc.). Each hook can implement 4 methods. The way they are called is demonstrated in the following snippet:
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()
    Here we will use a hook to calculate the validation loss after each step
    """
    def __init__(self, cfg):
        super().__init__() # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg)) # builds the dataloader from the provided cfg
        self.best_loss = float('inf') # Current best loss, initially infinite
        self.weights = None # Current best weights, initially none
        self.i=0 # Something to use for counting the steps

    def after_step(self): # after each step

        if self.trainer.iter % 100 == 0:
            print(f"Hello at iteration {self.trainer.iter}!") # print the current iteration if it's divisible by 100

        data = next(self._loader) # load the next piece of data from the dataloader

        with torch.no_grad(): # disables gradient calculation; we don't need it here because we're not training, just calculating the val loss
            loss_dict = self.trainer.model(data) # more about it in the next section

            losses = sum(loss_dict.values()) #
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced) # puts these metrics into the storage (where detectron2 logs metrics)

                # save best weights
                if losses_reduced < self.best_loss: # if current loss is lower
                    self.best_loss = losses_reduced # saving the best loss
                    self.weights = copy.deepcopy(self.trainer.model.state_dict()) # saving the best weights

def show_results(cfg, dataset_dicts, predictor, samples=10):

    for idx, data in enumerate(random.sample(dataset_dicts, samples)):
        img_path = data["file_name"]
        img = cv2.imread(img_path)

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("KITTI-MOTS_test"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        inference_img = v.get_image()[:, :, ::-1]
        cv2.imwrite(f'test{idx}.png', inference_img)