# comment for commiting

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, DataLoader
import numpy as np
import cv2

import pytorch_lightning as pl
IMG_EXT=set(['.png', '.jpg', '.jpeg', '.gif', '.tif'])
def _check_path(path) -> str:
    """
        Function to convert path object to string, if necessary
    """
    if isinstance(path, Path):
        path = str(path)
    return path

def load_label(fpath : str) -> np.ndarray:
    return load_image(fpath)

def load_image(fpath : str) -> np.ndarray:
    """
        Using PIL because OpenCV changes channel order!
    """
    img2 = Image.open(fpath)
    img2_np = np.array(img2)
    return img2_np

class ALGDataset(VisionDataset):
    
    def __init__(self, root: str, transforms = None, transform = None, target_transform = None,
                 img_folder : str = "images", label_folder = "labels",
                 num_classes : int = 3, img_ext = ".tif", label_ext=".tif"
                 ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.num_classes = num_classes
        self.img_dir = Path(self.root) / img_folder
        self.label_dir = Path(self.root) / label_folder

        self.img_ext = img_ext
        self.label_ext = label_ext

        # self.img_list = list(p.resolve().stem for p in self.img_dir.glob("**/*") if p.suffix in IMG_EXT)            # potentially replace by x.stem
        self.img_list = list([x.stem for x in self.img_dir.glob("*"+img_ext)])
        print("Test debug line")

    def __len__(self) -> int:
        return len(self.img_list)
    

    def __getitem__(self, idx: int) -> tuple:
            if torch.torch.is_tensor(idx):
                idx = idx.tolist()
            
            fname = self.img_list[idx]
            img_name = self.img_dir / (fname + self.img_ext)
            label_name = self.label_dir / (fname + self.label_ext)

            img = load_image(img_name)

            label = load_label(label_name)

            if self.transforms is not None:
                transformed = self.transforms(image=img, mask=label)
                label = transformed["mask"]
                img = transformed["image"]
            
            return img, label