import os
from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl

from generate_subdataset import crop_dataset
from train_autoencoder import train_autoencoder
from train_subensemble import train_subensemble
from inference_subensemble import inference_subensemble
from test_subensemble import test_subensemble
from alg.utils import get_subdirs, copy_img_and_label
from baseline_resnet import parse_resnet
from baseline_ae import parse_ae

def parse_subens(parser : ArgumentParser) -> ArgumentParser:
    # Required arguments
    parser.add_argument(
        "--sample",
        help="""Whether to use subensemble sampling or not - as weak / strong baseline""",
        action="store_true", 
    )

    # Whether to use an autoencoder in the first place
    parser.add_argument(
        "--autoenc",
        help="""Whether to run an Autoencoder or not""",
        action="store_true"
    )

    return parser

if __name__=="__main__":
    # setup of start
    np.random.seed(0)
    pl.seed_everything(0)

    # setup of arguments
    parser = parse_resnet()
    parser = parse_ae(parser)
    parser = parse_subens(parser)
    args = parser.parse_args()

    # default arguments
    name = args.name
    n_labeled = args.n_labeled
    epochs_labeled = args.epochs_labeled    
    resnet_version = args.resnet_version

    # Autoencder settings
    autoenc = args.autoenc
    n_unlabeled = args.n_unlabeled
    epochs_unlabeled = args.epochs_unlabeled
    denoising = args.denoising
    
    # setup of directories
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datadir = os.path.join(basedir, 'data', args.name)
    sites_basedir = os.path.expanduser("~/src/csu/data/ALG/sites")
    sites_dirs = [
        os.path.join(sites_basedir, "site1_McD"),
        os.path.join(sites_basedir, "site2_GC"),
        os.path.join(sites_basedir, "site3_Kuma"),
        os.path.join(sites_basedir, "site4_TSR")
    ]

    # start with Site 1 - unlabeled images + 100 labeled images
    if autoenc:
        site_1_baseraw = os.path.join(sites_dirs[0], 'raw')
        site1_rawdirs = get_subdirs(site_1_baseraw)
        raw_root = os.path.join(datadir, 'raw')
        raw_output = os.path.join(raw_root, 'images')
        crop_dataset(site1_rawdirs, n_unlabeled, raw_output)

    use_subensemble = args.sample # ! factor for strong baseline -> if false, will copy random images -> 
    base_logdir = os.path.join(basedir, 'lightning_logs', args.name, 'subensemble_pipeline' if use_subensemble else "baseline_select")
    if not autoenc:
        base_logdir = os.path.join(base_logdir, "no_ae")
    else:
        ldstr = "ae_denoise" if denoising else "ae"
        base_logdir = os.path.join(base_logdir, ldstr)

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

    copy_img_and_label(n_labeled, sites_dirs[0], labeled_output)
    # train_subensembles - return position 0 is the autoencoder path, the others are the heads
    model_settings = {
        "epochs" : epochs_labeled,          
        "num_classes" : 1,
        "optim" : "adam",
        "lr" : 1e-3,
        "bs" : 16,
        "resnet_version" : resnet_version   # 18 or 34
    }
    # se_logdir = os.path.join(base_logdir, site_name)
    # subens_paths = train_subensemble(autoencoder_path, se_logdir, labeled_output, model_settings)

    load_true = True
    for site in sites_dirs[1:]:
        site_name = os.path.basename(site)
        logdir = os.path.join(base_logdir, site_name)

        # generate new raw dataset
        if autoenc:
            print("Generating raw dataset for autoencoder training from site: {}".format(
                    site_name
                ))
            _rawdir = os.path.join(site, 'raw')
            input_rawdirs = get_subdirs(_rawdir)
            crop_dataset(input_rawdirs, n_unlabeled, raw_output)

            # train autoencoder - with previous data + "site"
            print("Completed copying dataset - Training autoencoder for site 0 and site: {}".format(
                site
            ))
            ae_logdir = os.path.join(logdir, "ae")
            autoenc_path = train_autoencoder(32, raw_root, ae_logdir, resnet_version=resnet_version, epochs_unlabeled=epochs_unlabeled, denoising=denoising)

            print("Completed training autoencoder - training subensemble heads on labeled dataset from sites previous to {}".format(
                site
            ))

        # train heads with dataset from sites-1 
        subens_paths = train_subensemble(autoenc_path if autoenc else None, logdir, labeled_output, model_settings)

        # inference subensemble on site
        site_p = os.path.join(site, "input_images") 
        print("Starting inference on site {} with models: {}.\nLogging to:{}".format(
            site_name, subens_paths, logdir
        ))
        res = test_subensemble(subens_paths, site, model_settings, logdir, img_folder="input_images", label_folder="mask_images", from_ae=autoenc)
        df = inference_subensemble(subens_paths, site, model_settings, logdir,
                                img_folder="input_images", label_folder="mask_images",
                                load_true=load_true)

        # sort the label files
        df.sort_values('entropy', inplace=True, ascending=False)
        label_names = df.index[:n_labeled]
        
        # calculate the accuracy for the binary case
        if load_true:            # or: if "label" in df.columns
            acc = (df["vote"] ==df["label"]).mean()
            print("Accuracy: for site: {}: {}".format(site_name, acc))

        # copy the files
        copy_img_and_label(label_names if use_subensemble else n_labeled, site, labeled_output)        