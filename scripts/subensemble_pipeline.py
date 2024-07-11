import os
from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl

from generate_subdataset import crop_dataset
from train_autoencoder import train_autoencoder
from train_subensemble import train_subensemble
from test_subensemble import test_subensemble
from alg.utils import copy_img_and_label, clean_directory
from baseline_resnet import parse_resnet, get_sites
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

    # How many heads to use
    parser.add_argument(
        "--heads",
        help="""How many heads to use. Defaults to 5""",
        type=int, default=5
    )

    return parser

if __name__=="__main__":
    # setup of arguments
    parser = parse_resnet()
    parser = parse_ae(parser)
    parser = parse_subens(parser)
    args = parser.parse_args()

    # seed setup
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)   
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
    datadir = os.path.join(basedir, 'data', name, args.datadir)
    sites_dirs, img_folder = get_sites()

    # start with Site 1 - unlabeled images + 100 labeled images
    if autoenc:
        site_1_rawdir = os.path.join(sites_dirs[0], img_folder)
        
        raw_root = os.path.join(datadir, 'raw')
        raw_output = os.path.join(raw_root, 'images')
        crop_dataset([site_1_rawdir], n_unlabeled, raw_output, seed=args.seed)

    use_subensemble = args.sample # ! factor for strong baseline -> if false, will copy random images -> 
    base_logdir = os.path.join(basedir, 'lightning_logs', name, 'subensemble_pipeline' if use_subensemble else "baseline_select")
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

    #! always start with 100 labeled and then add 20
    copy_img_and_label(100, sites_dirs[0], labeled_output, seed=args.seed)
    # train_subensembles - return position 0 is the autoencoder path, the others are the heads
    model_settings = {
        "epochs" : epochs_labeled,          
        "num_classes" : 1,
        "optim" : "adam",
        "lr" : 1e-3,
        "bs" : 32,
        "resnet_version" : resnet_version   # 18 or 34
    }
    # se_logdir = os.path.join(base_logdir, site_name)
    # subens_paths = train_subensemble(autoencoder_path, se_logdir, labeled_output, model_settings)
    if args.retrain: subens_paths = [None] # if retraining from the last one, then
    
    load_true = True
    for i, site in enumerate(sites_dirs[1:]):
        site_name = os.path.basename(site)
        logdir = os.path.join(base_logdir, site_name)
        prev_sitename = os.path.basename(sites_dirs[i-1])
        subens_logdir = os.path.join(base_logdir, prev_sitename)

        # generate new raw dataset
        if autoenc:
            print("Generating raw dataset for autoencoder training from site: {}".format(
                    site_name
                ))
            rawdir = os.path.join(site, img_folder)
            crop_dataset([rawdir], n_unlabeled, raw_output, seed=args.seed)

            # train autoencoder - with previous data + "site"
            print("Completed copying dataset - Training autoencoder for site: {}".format(
                site_name
            ))
            ae_logdir = os.path.join(logdir, "ae")
            autoenc_path = train_autoencoder(
                32,
                raw_root,
                ae_logdir,
                resnet_version=resnet_version, 
                epochs_unlabeled=epochs_unlabeled,
                denoising=denoising,
                use_fft=True,
                prev_model = subens_paths[0] if args.retrain else None
                )

            print("Completed training autoencoder")
            if args.retrain:
                print("Retraining activated. Cleaning raw image directory from png files")
                clean_directory(raw_output)
            
        print("Training subensemble heads on labeled dataset from sites {}".format(
                prev_sitename
            ))

        # train heads with dataset from sites-1 
        subens_paths = train_subensemble(autoenc_path if autoenc else None, subens_logdir, labeled_output, model_settings, n=args.heads, seed=args.seed)

        # inference subensemble on site
        site_p = os.path.join(site, "input_images") 
        print("Starting inference on site {} with models: {}.\nLogging to:{}".format(
            site_name, subens_paths, logdir
        ))
        res = test_subensemble("test", subens_paths, site, model_settings, logdir, img_folder="input_images", label_folder="mask_images", from_ae=autoenc, from_retrain=args.retrain, load_true=load_true)
        df = test_subensemble("inference", subens_paths, site, model_settings, logdir, img_folder="input_images", label_folder="mask_images", from_ae=autoenc,  from_retrain=args.retrain, load_true=load_true)

        # sort the label files
        df.sort_values('entropy', inplace=True, ascending=False)
        label_names = df.index[:n_labeled]
        
        # calculate the accuracy for the binary case
        if load_true:            # or: if "label" in df.columns
            acc = (df["vote"] ==df["label"]).mean()
            print("Accuracy: for site: {}: {}".format(site_name, acc))

        # copy the files
        copy_img_and_label(label_names if use_subensemble else n_labeled, site, labeled_output, seed=args.seed)        