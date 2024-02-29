# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
# lightning Changelog: https://lightning.ai/docs/pytorch/stable/upgrade/from_1_7.html
# tutorial: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
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
    for j in range(10000):
        fig,axs = plt.subplots(1,2, figsize=(16,9))
        i = np.random.randint(len(alg_ds))
        name = alg_ds.img_list[i]
        img, mask = alg_ds[i]
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[1].imshow(mask)
        axs[1].axis('off')
        plt.suptitle("Sample {}, img: {}".format(i, name))
        plt.show()

        