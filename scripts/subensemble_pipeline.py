import os
from pathlib import Path
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

def copy_img_and_label(n : int | list, input_basedir : str, output_basedir : str, i_imgs : str = "input_images", i_labels : str = "mask_images", o_imgs : str = "images", o_labels : str = "labels", fext : str = ".tif"):
    input_imgdir = Path(input_basedir) / i_imgs
    input_labeldir = Path(input_basedir) / i_labels
    
    output_imgdir = Path(output_basedir) / o_imgs
    output_labeldir = Path(output_basedir) / o_labels

    if isinstance(n, int):
        img_list = list([x.stem for x in input_imgdir.glob("*" + fext)])
        img_ids = np.random.choice(img_list, n)
    else:
        img_ids = n
    
    for img_id in img_ids:
        inp_img_f = input_imgdir / (img_id + fext)
        inp_lab_f = input_labeldir / (img_id + fext)
        outp_img_f = output_imgdir / (img_id + fext)
        outp_lab_f = output_labeldir / (img_id + fext)
        shutil.copy2(inp_img_f, outp_img_f)
        shutil.copy2(inp_lab_f, outp_lab_f)
    print("Copied {} images and associated label files from: {} to: {}".format(
        len(img_ids), input_basedir, output_basedir
    ))

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
    site_1_baseraw = os.path.join(sites_dirs[0], 'raw')
    site1_rawdirs = get_subdirs(site_1_baseraw)
    raw_root = os.path.join(datadir, 'raw')
    raw_output = os.path.join(raw_root, 'images')
    # crop_dataset(site1_rawdirs, 10, raw_output)

    # train autoencoder with unlabeled images
    base_logdir = os.path.join(basedir, 'lightning_logs', 'subensemble_pipeline')
    ae_logdir = os.path.join(base_logdir, 'ae')
    autoencoder_path = train_autoencoder(32, raw_root, ae_logdir)
    print("Trained Autoencoder at: {}".format(
        autoencoder_path
    ))
    autoencoder_paths = [autoencoder_path]

    # get subdataset for labeled heads training 
    labeled_output = os.path.join(datadir, 'labeled')
    labeled_imgs = os.path.join(labeled_output, 'images')
    labeled_labels = os.path.join(labeled_output, 'labels')    

    # copy_img_and_label(100, 
    #                    sites_dirs[0],
    #                    labeled_output,
    #                    )
    # train_subensembles - return position 0 is the autoencoder path, the others are the heads
    model_settings = {
        "epochs" : 10,          #! change back
        "num_classes" : 1,
        "optim" : "adam",
        "lr" : 1e-3,
        "bs" : 16
    }
    se_logdir = os.path.join(base_logdir, "se_{}".format(0))
    subens_paths = train_subensemble(autoencoder_path, se_logdir, labeled_output, model_settings)

    load_true = True
    for site in sites_dirs[1:]:
        print("Training subensemble for site {}".format(site))
        # inference subensemble
        logd = os.path.join(base_logdir, 'inference_{}'.format(site))
        site_p = os.path.join(site, "input_images") # todo - naming convention passthrough
        df = inference_subensemble(subens_paths, site, model_settings, logd,
                                   img_folder="input_images", label_folder="mask_images",
                                 load_true=load_true)

        # sort the label files
        df.sort_values('entropy', inplace=True, ascending=False)
        label_names = df.index[:20]
        
        # calculate the accuracy for the binary case
        if load_true:            # or: if "label" in df.columns
            acc = (df["vote"] ==df["label"]).mean()

        # copy the files
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
        
        