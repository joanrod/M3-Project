import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class MITSceneDataset(Dataset):
    """
    Function to initialize the dataset for the MIT-Scene
    """
    def __init__(self, image_dir):
        """
        Initialization of the Dataset Class
        :param image_dir: str of the folder where images are stored (either 'MIT_split/train' or 'MIT_split/test')
        """

        self.image_dir = image_dir

        # Search all the images in the root file and store the paths and their labels
        images = []
        labels = []
        for root, subdirs, files in os.walk(self.image_dir):            # Iterate through the root
            for subdir in subdirs:                                      # Iterate through the subdirectories
                subclass_imgs = os.listdir(os.path.join(root,subdir))   # Find all the names of the images
                images = images + subclass_imgs                         # Concatenate the names with the rest
                labels = labels + [subdir] * len(subclass_imgs)         # Concatenate the image label

        self.images = images
        self.labels = labels

    def __len__(self):
        """Length of the Dataset, len(MITSceneDataset)"""
        return len(self.images)

    def __getitem__(self, index):
        """
        Function to return the corresponding image and label of index
        :param index: int, position of the image to retrieve
        :return:
        """
        img_path = os.path.join(self.image_dir, self.labels[index], self.images[index])
        image_label = self.labels[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = image.transpose((2,0,1))

        return image, image_label

if __name__=="__main__":
    """
    Tests to see if there are errors
    """

    dataset = MITSceneDataset('MIT_split/train')                    # Dataset creation
    print(f"Length of the dataset: {len(dataset)} images")          # Length of the dataset
    image, label = dataset[0]                                       # Obtain image and label of the first position

    plt.imshow(image)                                               # Show image and label
    plt.title(label)
    plt.show()


