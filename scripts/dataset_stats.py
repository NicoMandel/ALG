import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from alg.dataloader import ALGDataset
from alg.utils import load_config

def count_histograms(alg_ds):
    # counting histograms
    percentage_ranges = range(0, 100, 1)
    counts = [0] * len(percentage_ranges)

    zero_percentages = np.array([(np.count_nonzero(mask == 0) / mask.size) * 100 for _, mask in alg_ds])
    perc_ranges = np.arange(0, 101, 10)
    hist, bin_edges = np.histogram(zero_percentages, bins=perc_ranges)

    # plt.bar(bin_edges[:-1], hist / len(alg_ds), width=8)
    plt.hist(zero_percentages, weights=np.ones(len(zero_percentages)) / len(zero_percentages), bins=range(0,101,10))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Percent of ALG pixels in image')
    plt.ylabel('Percentage of Images')
    plt.title('Histogram of Images with ALG')
    # plt.xticks(perc_ranges)
    plt.grid(axis='y')
    plt.show()

def test_label_loading(alg_ds):
    for img, label in alg_ds:
        plt.imshow(img)
        plt.title(label)
        plt.show()

def test_mask_loading(alg_ds):
    for img, label in alg_ds:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(img)
        axs[1].imshow(label)
        # plt.imshow(img)
        # plt.title(label)
        plt.show()


if __name__=="__main__":
    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    datadir = os.path.join(basedir, 'data')
    alg_ds = ALGDataset(datadir, threshold=0.75)

    confdir = os.path.join(basedir, 'config')
    conff = os.path.join(confdir, 'config.yml')
    cfg = load_config(conff)
    
    print(len(alg_ds))
    # count_histograms(alg_ds)
    test_label_loading(alg_ds)