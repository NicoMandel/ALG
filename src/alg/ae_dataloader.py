# comment for commiting

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, DataLoader
import numpy as np
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

IMG_EXT=set(['.png', '.jpg', '.jpeg', '.gif', '.tif', '.JPG', '.PNG'])

def _check_path(path) -> str:
    """
        Function to convert path object to string, if necessary
    """
    if isinstance(path, Path):
        path = str(path)
    return path

def load_image(fpath : str) -> np.ndarray:
    """
        According to: https://albumentations.ai/docs/getting_started/image_augmentation/
    """
    img2 = Image.open(fpath)
    return img2

class ALGRAWDataset(VisionDataset):
    
    def __init__(self, root: str, transforms = None, transform = None, target_transform = None, 
                 img_folder : str = "images", load_names : bool = False
                 ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.img_dir = Path(self.root) / img_folder
        self.load_names = load_names
        
        self.img_list = []
        for ext in IMG_EXT:
            nf = self.img_dir.glob("*" + ext)
            fns = list([fn.name for fn in nf])
            self.img_list.extend(fns)
        
    def __len__(self) -> int:
        return len(self.img_list)
    

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, np.ndarray]:
            if torch.torch.is_tensor(idx):
                idx = idx.tolist()
            
            fname = self.img_list[idx]

            # image loading
            img_name = self.img_dir / fname
            if self.load_names:
                return img_name
            
            img = load_image(img_name)

            if self.transforms is not None:
                img = self.transforms(img)

            return img, 1
    
class ALGRAWDataModule(pl.LightningDataModule):
    """
        Datamodule to split for training and validation
    """

    def __init__(self, root : str, img_folder : str = "images",
                 transforms  = None, val_percentage : float = 0.2,
                 num_workers : int = 4, batch_size : int = 16) -> None:
        super().__init__()
        self.root = root
        self.img_folder = img_folder

        self.transforms = transforms
        self.val_percentage = val_percentage
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.default_dataset = ALGRAWDataset(self.root, self.transforms, img_folder=self.img_folder)


    def prepare_data(self):
        """
            Preparing data splits
        """
        
        # Splitting the dataset
        dataset_len = len(self.default_dataset)
        train_part = int( (1-self.val_percentage) * dataset_len)
        val_part = dataset_len - train_part

        # Actual datasets
        self.train_dataset, self.val_dataset = random_split(self.default_dataset, [train_part, val_part])

    # Dataloaders:
    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        pin_memory=True,
        )
        return dl
    
    def val_dataloader(self) -> DataLoader:
        dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                        drop_last=False
        # pin_memory=True
        )
        return dl
    
