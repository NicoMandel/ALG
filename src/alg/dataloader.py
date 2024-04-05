# comment for commiting

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, DataLoader
import numpy as np
import cv2
import albumentations as A
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
        tif loaded as RGBA -> needs conaversion to RGb
    """
    img2 = Image.open(fpath).convert('RGB')
    img2_np = np.array(img2)
    return img2_np

def load_txt(fpath : str) -> float:
    with open(fpath, 'r') as f:
        val = float(f.read().rstrip())
    return val

class ALGDataset(VisionDataset):
    
    def __init__(self, root: str, transforms = None, transform = None, target_transform = None,
                 img_folder : str = "images", label_folder = "labels",
                 num_classes : int = 3, img_ext = ".tif", label_ext=".tif",
                 threshold : float = None
                 ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.num_classes = num_classes
        self.img_dir = Path(self.root) / img_folder
        self.label_dir = Path(self.root) / label_folder

        self.img_ext = img_ext
        self.label_ext = label_ext
        if label_ext.lower() in IMG_EXT:
            self._isimg = True
            self.label_load_fn = load_label
        elif label_ext.lower() == ".txt":
            self.label_load_fn = load_txt
            self._isimg = False
        else:
            raise ValueError("Unknown Label Extension")

        self.threshold = threshold

        # self.img_list = list(p.resolve().stem for p in self.img_dir.glob("**/*") if p.suffix in IMG_EXT)            # potentially replace by x.stem
        self.img_list = list([x.stem for x in self.img_dir.glob("*"+img_ext)])

    def __len__(self) -> int:
        return len(self.img_list)
    

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
            if torch.torch.is_tensor(idx):
                idx = idx.tolist()
            
            fname = self.img_list[idx]

            # image loading
            img_name = self.img_dir / (fname + self.img_ext)
            img = load_image(img_name)

            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]

            # Label Loading
            label_name = self.label_dir / (fname + self.label_ext)
            label = self.label_load_fn(label_name)
            if self.threshold is not None:
                if self._isimg:
                    label = self._clean_mask(mask=label)
                label = self._convert_label(label)

            return img, label
    
    def _clean_mask(self, mask : np.ndarray) -> np.ndarray:
        """
            Function to clean loading artifacts from mask, when values are larger than 0, they get allocated to 127
        """
        mask[(mask > 0) & (mask < 255)] = 127
        return mask

    def _convert_label(self, label : np.ndarray | float) -> np.ndarray:
        if isinstance(label, np.ndarray):
            return 1 if ((np.count_nonzero(label == 0) / label.size) > self.threshold) else 0
        else:
            return 1 if label > self.threshold else 0

class ALGDataModule(pl.LightningDataModule):
    """
        Datamodule to split for training and validation
    """

    def __init__(self, root : str, img_folder : str = "images", label_folder : str = "labels",
                 num_classes : int = 3, img_ext = ".tif", label_ext=".tif",
                 clean_values : tuple = (0, 127), threshold : float = 0.6,
                 transforms : A.Compose = None, val_percentage : float = 0.2,
                 num_workers : int = 4, batch_size : int = 16) -> None:
        super().__init__()
        self.root = root
        self.img_folder = img_folder
        self.img_ext = img_ext
        self.label_folder = label_folder
        self.label_ext = label_ext

        self.num_classes = num_classes
        self.clean_values = clean_values
        self.threshold = threshold

        self.transforms = transforms
        self.val_percentage = val_percentage
        
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        """
            Preparing data splits
        """
        self.default_dataset = ALGDataset(self.root, self.transforms, img_folder=self.img_folder, label_folder=self.label_folder,
                                            img_ext=self.img_ext, label_ext=self.label_ext,
                                            threshold=self.threshold)
        
        # Splitting the dataset
        dataset_len = len(self.default_dataset)
        train_part = int( (1-self.val_percentage) * dataset_len)
        val_part = dataset_len - train_part

        # Actual datasets
        self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])

    # Dataloaders:
    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        # pin_memory=True
        )
        return dl
    
    def val_dataloader(self) -> DataLoader:
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        # pin_memory=True
        )
        return dl
    
