import os
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

from alg.model import ResNetClassifier
from alg.utils import copy_img_and_label
from train_model import train_model
from test_model import test_model

if __name__=="__main__":
    epochs_labeled = 200     #! change back to 200
    n_labeled = 100

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

    base_logdir = os.path.join(basedir, 'lightning_logs', "eccv", 'baseline_resnet')

    # get subdataset for labeled heads training 
    labeled_output = os.path.join(datadir, 'labeled')

    # use all labels here for training!
    input_imgdir = Path(sites_dirs[0]) / "input_images"
    img_list = list([x.stem for x in input_imgdir.glob("*" + ".tif")])
    copy_img_and_label(n_labeled, sites_dirs[0], labeled_output)  # ! img_list
    model_settings = {
        "num_epochs" : epochs_labeled,         
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
    model = ResNetClassifier(
        num_classes=model_settings["num_classes"],
        resnet_version=model_settings["model_version"],
        optimizer=model_settings["optim"],
        lr=model_settings["lr"],
        batch_size=model_settings["bs"],
        transfer=model_settings["transfer"],
        tune_fc_only=model_settings["tune_fc_only"],
    )

    model_p = None
    load_true = True
    for site in sites_dirs[1:]:
        
        site_name = os.path.basename(site)
        if model_p is not None:
            model = ResNetClassifier.load_from_checkpoint(model_p)

        # train model with labeled dataset from sites-1 
        model_logdir = os.path.join(base_logdir, site_name)
        model_p, logger = train_model(model, model_settings, "resnet_{}".format(model_settings["model_version"]),  model_logdir, labeled_output)

        # test model on new site
        site_p = os.path.join(site, "input_images") 
        print("Starting test on site {} with model: {}.\nLogging to:{}".format(
            site_name, model_p, logger.save_dir
        ))

        res = test_model(model_p, site, threshold=model_settings["threshold"], logger=logger,
                    img_folder="input_images", label_folder="mask_images", fext=".tif")
        
        # calculate the accuracy for the binary case
        if load_true:            # or: if "label" in df.columns
            acc = res[0]["test_acc_epoch"]
            print("Accuracy: for site: {}: {}".format(site_name, acc))

        # use all labels here for training!
        input_imgdir = Path(site) / "input_images"
        img_list = list([x.stem for x in input_imgdir.glob("*" + ".tif")])
        copy_img_and_label(n_labeled, site, labeled_output)   # ! img_list        