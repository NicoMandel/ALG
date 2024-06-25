import os
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms as torch_tfs
from torch.utils.data import random_split, DataLoader, Subset

from alg.dataloader import ALGDataset
from generate_subdataset import crop_dataset
from train_autoencoder import train_autoencoder
from train_subensemble import train_subensemble
from inference_subensemble import inference_subensemble

def get_subdirs(dirname : str) -> list[str]:
    return [os.path.join(dirname, name) for name in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, name))]

if __name__=="__main__":
    np.random.seed(0)
    pl.seed_everything(0)

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datadir = os.path.join(basedir, 'data', 'reduced_example')
    sites_basedir = os.path.expanduser("~/src/csu/data/ALG/sites")
    sites_dirs = [
        os.path.join(sites_basedir, "site1_McD"),
        os.path.join(sites_basedir, "site2_GC"),
        os.path.join(sites_basedir, "site3_Kuma"),
        os.path.join(sites_basedir, "site4_TSR")
    ]
    # start with Site 1 - unlabeled images + 100 labeled images
    site1_crops_dir = os.path.join(datadir, 'site1_crops')
    site_1_baseraw = os.path.join(sites_dirs[0], 'raw')
    site1_rawdirs = get_subdirs(site_1_baseraw)
    raw_output = os.path.join(datadir, 'raw')
    if not os.path.exists(raw_output):
        os.makedirs(raw_output)
    crop_dataset(site1_rawdirs, 10, raw_output)

    # train autoencoder with unlabeled images
    base_logdir = os.path.join(basedir, 'lightning_logs', 'subensemble_pipeline')
    ae_logdir = os.path.join(base_logdir, 'ae')
    autoencoder_path = train_autoencoder(32, raw_output, ae_logdir)
    print("Trained Autoencoder at: {}".format(
        autoencoder_path
    ))
    autoencoder_paths = [autoencoder_path]

    # get subdataset for labeled heads training 
    labeled_output = os.path.join(datadir, 'labeled')
    if not os.path.exists(labeled_output):
        os.makedirs(labeled_output)

    # init a base dataset
    mean = (0.5,)
    std = (0.5,)
    tfs = torch_tfs.Compose([
        torch_tfs.ConvertImageDtype(torch.float),
        torch_tfs.Normalize(mean, std)        
    ])
    base_ds = ALGDataset(
        root=sites_dirs[0],
        img_folder="input_images",
        label_folder="mask_images",
        transforms=tfs,
        num_classes=1,
        threshold=0.5
    )
    # Subsampling the dataset to only hold some percentage of training data - low volume!
    ds_count = 100
    ds_inds = np.random.choice(len(base_ds), ds_count)
    site_1_ds = Subset(base_ds, ds_inds)

    # train_subensembles - return position 0 is the autoencoder path, the others are the heads
    model_settings = {
        "epochs" : 200,
        "num_classes" : 1,
        "optim" : "adam",
        "lr" : 1e-3,
        "bs" : 16
    }
    se_logdir = os.path.join(base_logdir, "se_{}".format(0))
    subens_paths = train_subensemble(autoencoder_path, se_logdir, site_1_ds, model_settings)

    for site in sites_dirs[1:]:
        # inference subensemble
        logd = os.path.join(base_logdir, 'inference_{}'.format(site))
        df = inference_subensemble(subens_paths, site, model_settings, logd)

        #! in df we have the vote -> we can use this as categorical cross-entropy vs. training a simple model
        # Todo - put into test step of the model
        # sort the label files
        df.sort_values('entropy', inplace=True, ascending=False)
        label_names = df.index[:20]
        
        # copy the label files
        img_dir = os.path.join(site, 'input_images')
        label_dir = os.path.join(site, 'mask_images')
        for ln in label_names:
            imgf = os.path.join(img_dir, ln+'.tif')
            label_img = os.path.join(label_dir, ln+'.tif')
            shutil.copy2(imgf, target_imgs)
            shutil.copy2(label_img, target_labels)

        # TODO - generate raw dataset
        crop_dataset(input_rawdir, 10, crop_outputs)
        
        # TODO - retrain autoencoder
        autoenc_path = train_autoencoder(32, crop_outputs, log_directory)
         
        # retrain heads with new dataset
        ds = ALGDataset(crop_outputs)
        subens_paths = train_subensemble(autoenc_path, log_directory, ds, model_stgs, n=5)
        
        