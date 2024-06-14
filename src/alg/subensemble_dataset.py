# comment for commiting

import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import numpy as np
import albumentations as A
import pytorch_lightning as pl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from alg.dataloader import IMG_EXT, load_label, load_tif, load_txt, _clean_mask, _convert_label

class SubensembleDataset(VisionDataset):
    
    def __init__(self, root: str, transforms = None, transform = None, target_transform = None,
                 img_folder : str = "images", label_folder = "labels",
                 num_classes : int = 3, img_ext = ".tif", label_ext=".tif",
                 threshold : float = None, load_names : bool = False
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
        self.load_names = load_names

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
            img = load_tif(img_name)

            if self.transforms is not None:
                if isinstance(self.transforms, A.Compose):
                    transformed = self.transforms(image=img)
                    img = transformed["image"]
                else:
                    img = torch.as_tensor(np.array(img, copy=True))
                    img = img.permute((2, 0, 1))
                    img = self.transforms(img)

            # Label Loading
            label_name = self.label_dir / (fname + self.label_ext)

            # if loading names only - used for cleaning
            label = self.label_load_fn(label_name)
            if self.threshold is not None:
                if self._isimg:
                    label = _clean_mask(mask=label)
                label = _convert_label(label, self.threshold)

            if self.load_names:
                label = (label, fname)

            return img, label