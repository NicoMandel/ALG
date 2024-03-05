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
    ctr = 0
    for i in range(len(alg_ds)):
        name = alg_ds.img_list[i]
        img, mask, ident = alg_ds[i]
        unq = np.unique(mask)
        if unq.size > 3:
            ctr += 1
            print("image: {}\nUnique element count: {}".format(
                ident, len(unq)
            ))
            # fig, axs = plt.subplots(1,2)
            # axs[0].imshow(img)
            # # axs[0].xticks('off')
            # # axs[1].yticks('off')
            # # axs[0].yticks('off')
            # # axs[1].xticks('off')
            # axs[1].imshow(mask)
            # plt.suptitle(ident)
            # plt.show()
    print("{} images with blanks. Removed".format(ctr))