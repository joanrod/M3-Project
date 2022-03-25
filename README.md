# M5: Visual Recognition - Team 2
| Members | Contact | GitHub |
| :---         |   :---    |   :---    |
| Igor Ugarte Molinet | igorugarte.cvm@gmail.com | [igorugarteCVM](https://github.com/igorugarteCVM) | 
| Juan Antonio Rodríguez García | juanantonio.rodriguez@upf.edu  | [joanrod](https://github.com/joanrod) |
| Francesc Net Barnès | francescnet@gmail.com  | [cesc47](https://github.com/cesc47) |
| David Serrano Lozano | 99d.serrano@gmail.com | [davidserra9](https://github.com/davidserra9) |

---
## Week1
Image classification using Pytorch, available at M5/week1.

Slides for week 1: [Slides](https://docs.google.com/presentation/d/1FGRrmjkltlC7GpD8WeX_9TiXb5x-T6QKmyFojb2Qg8w/edit?usp=sharing)

To execute the program which trains and evaluates the model, run the following command:
```
$ python week1/train.py
```
## Week2
The main task for this week was to use the Faster RCNN [1] (Object Detection algorithm) and Mask RCNN [2] (Instance Segmentation algorithm) in the KITTI-MOTS [3] dataset using Detectron2 [4] to detect both pedestrians and cars.

![Object Detection](/week2/inference/0013.gif)

The model were pretrained on the Microsoft-COCO [5] dataset and finetunned on 8 sequences of the KITTI-MOTS dataset. All the reasoning and procedures can be seen in:

Slides for week 2: [Slides](https://docs.google.com/presentation/d/1ERkqOnMB56ElYuvYg9izsTqWn5aO28IjpjF5gZBFMKM/edit#slide=id.g11d85991502_0_90)

To fine-tune the models for the KITTI-MOTS dataset and evaluate the results run:
```
$ python week2/task_e.py
```

## Week3
The context of an image encapsulates rich information about how natural scenes and objects are related to each other. That is why, the task for this week was to explore how the Object Detection algorithms use context detect some object or improve some detections.

![PlaneAsBird](/week3/readmeimage.png)
The "Out-Of-Context" [6] dataset was used as well as custom images modified from the COCO dataset based on the Elephant in a room paper [7]. All the reasoning and procedured can be seen in:

Slides for week 3: [Slides](https://docs.google.com/presentation/d/1Hvv0NIu_j9Rd1Bp6VtYcSG42W5-3KcmsYLNb5G3_tno/edit?usp=sharing)

To run each section explained in the previos slides run:
```
$ python week3/task_{id_task}.py
```

The CVPR paper corresponding to all the tasks and experiments devoted to Object Detection and Instance Segmentation can be seen in: [Paper](https://www.overleaf.com/read/hcbxsbkrsmcb)

## References
[1] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).

[2] He, Kaiming, et al. "Mask r-cnn." Proceedings of the IEEE international conference on computer vision. 2017.

[3] Voigtlaender, Paul, et al. "Mots: Multi-object tracking and segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[4]Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick, [Detectron2](https://github.com/facebookresearch/detectron2), 2019.

[5] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.

[6] Choi, Myung Jin, Antonio Torralba, and Alan S. Willsky. "Context models and out-of-context objects." Pattern Recognition Letters 33.7 (2012): 853-862.

[7] Rosenfeld, Amir, Richard Zemel, and John K. Tsotsos. "The elephant in the room." arXiv preprint arXiv:1808.03305 (2018).
