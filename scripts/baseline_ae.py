import os
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

from generate_subdataset import crop_dataset
from train_autoencoder import train_autoencoder
from alg.utils import get_subdirs, copy_img_and_label
from train_from_ae import train_resnet_from_ae
from test_model import test_model

if __name__=="__main__":
    np.random.seed(0)
    pl.seed_everything(0)

    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datadir = os.path.join(basedir, 'data', "eccv")
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
    crop_dataset(site1_rawdirs, 10, raw_output)

    # train autoencoder with unlabeled images
    base_logdir = os.path.join(basedir, 'lightning_logs', "eccv", 'baseline_ae')
    # site_name = os.path.basename(sites_dirs[0])
    # ae_logdir = os.path.join(base_logdir, site_name, "ae")
    # autoencoder_path = train_autoencoder(32, raw_root, ae_logdir)
    # print("Trained Autoencoder at: {}".format(
    #     autoencoder_path
    # ))
    # autoencoder_paths = [autoencoder_path]

    # get subdataset for labeled heads training 
    labeled_output = os.path.join(datadir, 'labeled')
    # labeled_imgs = os.path.join(labeled_output, 'images')
    # labeled_labels = os.path.join(labeled_output, 'labels')    

    copy_img_and_label(100, sites_dirs[0], labeled_output)
    model_settings = {
        "num_epochs" : 5,          #! change back to 200
        "model_version" : 18,
        "num_classes" : 1,
        "optim" : "adam",
        "lr" : 1e-3,
        "bs" : 16,
        "transfer" : True,
        "threshold" : 0.5,
        "tune_fc_only" : False,
        "limit" : None
    }

    load_true = True
    for site in sites_dirs[1:]:
        site_name = os.path.basename(site)
        print("Generating raw dataset for autoencoder training from site: {}".format(
            site_name
        ))

        # generate new raw dataset
        _rawdir = os.path.join(site, 'raw')
        input_rawdirs = get_subdirs(_rawdir)
        crop_dataset(input_rawdirs, 10, raw_output)

        # train autoencoder - with previous data + "site"
        print("Completed copying dataset - Training autoencoder for site 0 and site: {}".format(
            site
        ))
        ae_logdir = os.path.join(base_logdir, site_name, "ae")
        autoenc_path = train_autoencoder(32, raw_root, ae_logdir)
        print("Completed training autoencoder - training model on labeled dataset from sites previous to {}".format(
            site
        ))

        # train model with labeled dataset from sites-1 
        model_logdir = os.path.join(base_logdir, site_name)
        model_p, logger = train_resnet_from_ae(autoenc_path,
                                        model_logdir,
                                        model_settings,
                                        labeled_output
                                        )

        # test model on new site
        logd = os.path.join(base_logdir,"inference", site_name)
        site_p = os.path.join(site, "input_images") 
        print("Starting test on site {} with model: {}.\nLogging to:{}".format(
            site_name, model_p, logd
        ))

        res = test_model(model_p, site, threshold=model_settings["threshold"], logger=logger,
                    img_folder="input_images", label_folder="mask_images", fext=".tif")
        
        # calculate the accuracy for the binary case
        if load_true:            # or: if "label" in df.columns
            acc = res["test_acc_epoch"]
            print("Accuracy: for site: {}: {}".format(site_name, acc))

        # use all labels here for training!
        input_imgdir = Path(site) / "input_images"
        img_list = list([x.stem for x in input_imgdir.glob("*" + ".tif")])
        copy_img_and_label(img_list, site, labeled_output)        