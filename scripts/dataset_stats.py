import os.path
import numpy as np
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from copy import deepcopy
import torchvision.transforms as torchtfs

import albumentations as A
from albumentations.pytorch import ToTensorV2

from alg.ae_dataloader import ALGRAWDataset, load_image
from alg.dataloader import ALGDataset
from alg.utils import load_config
from alg.ae_utils import NormalizeInverse

def count_histograms(alg_ds, ds_title = ""):
    # counting histograms
    percentage_ranges = range(0, 100, 1)
    counts = [0] * len(percentage_ranges)

    zero_percentages = np.array([(np.count_nonzero(mask == 0) / mask.size) * 100 for _, mask in alg_ds])
    perc_ranges = np.arange(0, 101, 10)
    hist, bin_edges = np.histogram(zero_percentages, bins=perc_ranges)

    # plt.bar(bin_edges[:-1], hist / len(alg_ds), width=8)
    plt.figure()
    plt.hist(zero_percentages, weights=np.ones(len(zero_percentages)) / len(zero_percentages), bins=range(0,101,10))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Percent of ALG pixels in image')
    plt.ylabel('Percentage of Images')
    plt.title('Histogram of Images with ALG in Dataset {}'.format(ds_title))
    # plt.xticks(perc_ranges)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("{}_stats.png".format(ds_title))
    # plt.show()

def test_label_loading(alg_ds):
    for i, (img, label) in enumerate(alg_ds):
        plt.imshow(img)
        plt.title(f"{alg_ds.img_list[i]}: {label}")
        plt.show()

def test_mask_loading(alg_ds):
    fig, axs = plt.subplots(1,2)
    for img, label in alg_ds:
        axs[0].imshow(img)
        axs[1].imshow(label)
        # plt.imshow(img)
        # plt.title(label)
        # plt.savefig('someimg.png')
        plt.show()

def clean_images(alg_ds, clean : bool = False, clean_value : int = 255):
    import os
    ctr = 0
    for img_name, mask_name in tqdm(alg_ds):
        img = load_image(img_name)
        if np.all(img == clean_value):
            ctr += 1
            if clean:
                os.remove(os.path.abspath(img_name))
                os.remove(os.path.abspath(mask_name))
            else:
                print(os.path.abspath(img_name))
                # print(os.path.abspath(mask_name)) 
            # print(alg_ds.img_list[i])
    print("Length of Dataset: {}. Pure {} images: {}, {:.2f}%".format(len(alg_ds), clean_value, ctr, ctr / len(alg_ds) * 100))

def test_raw_dataset(basedir : str, save = False):
    """
        Function to test the raw dataset dataloading
    """
    np.random.seed(1)
    datadir = os.path.join(basedir, 'data', 'raw')
    tfs = A.Compose([
        A.RandomCrop(32,32),
        ToTensorV2(always_apply=True)
    ])
    # post_tfs_norm = A.Compose([
    #     # A.Normalize((0.5,), (0.5,)),
    #     ToTensorV2(always_apply=True)
    # ])
    post_tf_norm = torchtfs.Normalize((0.5,) , (0.5,))
            
    raw_ds = ALGRAWDataset(datadir, transforms=tfs)
    
    unnormalize = NormalizeInverse((0.5,) , (0.5,))
    fig, axs = plt.subplots(1,2)
    for i, (img, label) in enumerate(raw_ds):
        img_norm = deepcopy(img)
        img = img.permute(1,2,0).int()

        img_norm = post_tf_norm(img_norm.float())   # .float()
        img_norm =img_norm.permute(1,2,0)   # .int()
        img_norm = unnormalize(img_norm).int()
        
        # img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        # img_norm = img_norm.astype(int)


        axs[0].imshow(img)
        axs[0].set_title("unnormalized")
        axs[1].imshow(img_norm)
        axs[1].set_title("normalized")
        plt.suptitle(f"{raw_ds.img_list[i]}: {label}")
        if not save:
            plt.show()
        else:
            plt.savefig('someimg.png')

def clean_raw_dataset(raw_ds, clean : bool = False, img_size : int = 256):
    import os
    ctr = 0
    for img_name in tqdm(raw_ds):
        img = load_image(img_name)
        if img.size != (img_size, img_size):
            ctr += 1
            if clean:
                os.remove(os.path.abspath(img_name))
            else:
                print(f"{img_name} not ({img_size} x {img_size}), but {img.size}")
    print(f"Length of Dataset: {len(raw_ds)}. Images not of size: ({img_size} x {img_size}): {ctr}, {ctr / len(raw_ds) * 100:.2f}%")

if __name__=="__main__":
    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    dir_256 = os.path.join(basedir, 'data', '256')
    raw_ds = ALGRAWDataset(dir_256, load_names=True)
    clean_raw_dataset(raw_ds, img_size=256)
    test_raw_dataset(basedir, save=True)
    # Dataset Cleaning Block!
    dd_b = "~/src/csu/data/ALG/sites"
    ks = ["site1_McD", "site2_GC", "site3_Kuma", "site4_TSR"]
    dd_dict = {}
    for k in ks:
        ddir = os.path.join(dd_b, k)
        alg_ds = ALGDataset(ddir, threshold=None, img_folder="input_images", label_folder="mask_images")
        count_histograms(alg_ds, ds_title=k)
    datadir_site1 = "~/src/csu/data/ALG/sites/site1_McD"
    datadir_site2 = "~/src/csu/data/ALG/sites/site2_GC"
    datadir_site3 = "~/src/csu/data/ALG/sites/site3_Kuma"
    datadir_site4 = "~/src/csu/data/ALG/sites/site4_TSR"
    datadirs = [datadir_site1, datadir_site2, datadir_site3, datadir_site4]
    alg_ds = ALGDataset(datadir_site4, threshold=None,img_folder="input_images", label_folder="mask_images", load_names=True)
    # clean_images(alg_ds, clean=False, clean_value=0)

    datadir = os.path.join(basedir, 'data')
    alg_ds = ALGDataset(datadir, threshold=None)

    testdir = os.path.join(datadir, 'test')
    alg_test_ds = ALGDataset(
        root=testdir,
        label_ext=".txt",
        threshold=0.5
    )
    # test_label_loading(alg_test_ds)


    confdir = os.path.join(basedir, 'config')
    conff = os.path.join(confdir, 'config.yml')
    cfg = load_config(conff)
    
    print(len(alg_ds))
    # count_histograms(alg_ds)
    # test_label_loading(alg_ds)
    test_mask_loading(alg_ds)