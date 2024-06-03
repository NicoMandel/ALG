from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import numpy as np
from alg.ae_dataloader import load_image, IMG_EXT
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        help="""Directories to look for files""",
        action="append",
    )
    parser.add_argument(
        "-n",
        help="""Number of crops to generate""",
        type=int,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="""Output location of the files. Default is none, then will plot""",
        default=None
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        help="""Size of crops to generate. Defaults to 256""",
        default=256
    )
    parser.add_argument(
        "-e", "--extension",
        help="""File extension to use for output images. Defaults to png""",
        type=str,
        default="png"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    fdirs = {}
    for input_dir in args.input:
        ind = Path(input_dir)
        img_list = []
        for ext in IMG_EXT:
            nf = ind.glob("*" + ext)
            fns = list([fn.name for fn in nf])
            img_list.extend(fns)
        fdirs[input_dir] = img_list
        print("Found {} images in directory: {}".format(len(img_list), input_dir))
    
    key_arr = np.array(list(fdirs.keys()))
    for i in range(args.n):
        k_c = np.random.choice(key_arr)
        f_la = np.array(fdirs[k_c])
        f_c = np.random.choice(f_la)

        imf = Path(k_c) / f_c
        img = load_image(imf)
        w, h = img.size
        top = np.random.randint(0, h-args.size)
        left = np.random.randint(0, w-args.size)
        crop = img.crop((left, top, left+args.size, top+args.size))
        print(k_c)
        print(crop.size)
        cropname = f"{Path(f_c).stem}_{top}_{left}" + "." + args.extension
        if args.output:
            pass
        else:
            plt.imshow(crop)
            plt.title(cropname)
            plt.show()

    