import os.path
import numpy as np

from alg.dataloader import ALGDataset
from alg.utils import load_config

if __name__=="__main__":
    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    datadir = os.path.join(basedir, 'data')
    alg_ds = ALGDataset(datadir)

    confdir = os.path.join(basedir, 'config')
    conff = os.path.join(confdir, 'config.yml')
    cfg = load_config(conff)
    
    print(len(alg_ds))
    for i in range(len(alg_ds)):
        name = alg_ds.img_list[i]
        img, mask = alg_ds[i]
        print("Test Debug line to figure out file sizes and types")