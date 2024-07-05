import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import pytorch_lightning as pl

from alg.model import ResNetClassifier
from alg.utils import copy_img_and_label
from train_model import train_model
from test_model import test_model


def parse_resnet() -> ArgumentParser:
    parser = ArgumentParser("Script Setup")

    parser.add_argument(
        "name",
        help="""name of experiment - is used as subfolder in data/ and lightning_logs/""", type=str
        )
    # number of images
    parser.add_argument(
        "--n_labeled",
        help="""Number of labeled images to copy""",
        type=int, default=None
    )

    # epochs to run
    parser.add_argument(
        "--epochs_labeled",
        help="""Number epochs to run on labeled samples""",
        type=int, default=200
    )

    # Which resnet version - 18 or 34
    parser.add_argument(
        "--resnet_version",
        help="""Which resnet version to use - 18 or 34. defaults to 18""",
        type=int, default=18
    )

    # baseline seed
    parser.add_argument(
        "--seed",
        help="""Which seeed to start off from. Defaults to 0""", type=int, default=0
    )

    parser.add_argument(
        "--full", action="store_true",
        help="""Whether to use labeled samples or the full site"""
    )

    return parser

if __name__=="__main__":

    # setup arguments
    parser = parse_resnet()
    args = parser.parse_args()
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)
    epochs_labeled = args.epochs_labeled     #! change back to 200
    n_labeled = args.n_labeled
    resnet_version = args.resnet_version
    
    name =args.name
    # setup directories
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datadir = os.path.join(basedir, 'data', args.name)
    sites_basedir = os.path.expanduser("~/src/csu/data/ALG/sites")
    sites_dirs = [
        os.path.join(sites_basedir, "site1_McD"),
        os.path.join(sites_basedir, "site2_GC"),
        os.path.join(sites_basedir, "site3_Kuma"),
        os.path.join(sites_basedir, "site4_TSR")
    ]

    v_name = "baseline_resnet"
    if args.full: v_name += "_full"
    base_logdir = os.path.join(basedir, 'lightning_logs', args.name, v_name)

    # get subdataset for labeled heads training 
    labeled_output = os.path.join(datadir, 'labeled')

    # use labels here for training!
    if args.full:
        input_imgdir = Path(sites_dirs[0]) / "input_images"
        img_list = list([x.stem for x in input_imgdir.glob("*" + ".tif")])
        copy_img_and_label(img_list, sites_dirs[0], labeled_output) 
    else:
        copy_img_and_label(100, sites_dirs[0], labeled_output) 

    model_settings = {
        "num_epochs" : epochs_labeled,         
        "model_version" : resnet_version,
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
        model_p, logger = train_model(model, model_settings, "resnet_{}".format(model_settings["model_version"]),  model_logdir, labeled_output, seed=args.seed)

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
        if args.full:
            input_imgdir = Path(site) / "input_images"
            img_list = list([x.stem for x in input_imgdir.glob("*" + ".tif")])
            copy_img_and_label(n_labeled, site, labeled_output)
        else:
            copy_img_and_label(n_labeled, site, labeled_output)           