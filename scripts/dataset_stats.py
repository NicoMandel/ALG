import os.path
import numpy as np
import matplotlib.pyplot as plt

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
    
    # counting histograms
    percentage_ranges = range(0, 100, 10)
    counts = [0] * len(percentage_ranges)

    zero_percentages = np.array([(np.count_nonzero(mask == 0) / mask.size) * 100 for _, mask in alg_ds])
    perc_ranges = np.arange(0, 101, 10)
    hist, bin_edges = np.histogram(zero_percentages, bins=perc_ranges)

    plt.bar(bin_edges[:-1], hist / len(alg_ds), width=8)
    plt.xlabel('Percentage Range')
    plt.ylabel('Percentage of Images')
    plt.title('Histogram of Images with ALG')
    plt.xticks(perc_ranges)
    plt.grid(axis='y')
    plt.show()

    