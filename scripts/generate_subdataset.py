from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from PIL.Image import Image
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


def crop_dataset(input_dirs : list, n : int, output_dir : str, crop_size : int = 256, extension : str = "png") :
    fdirs = {}
    for input_dir in input_dirs:
        ind = Path(input_dir)
        img_list = []
        for ext in IMG_EXT:
            nf = ind.glob("*" + ext)
            fns = list([fn.name for fn in nf])
            img_list.extend(fns)
        fdirs[input_dir] = img_list
        print("Found {} images in directory: {}".format(len(img_list), input_dir))
    
    key_arr = np.array(list(fdirs.keys()))
    for i in tqdm(range(n), leave=True):
        k_c = np.random.choice(key_arr)
        f_la = np.array(fdirs[k_c])
        f_c = np.random.choice(f_la)

        imf = Path(k_c) / f_c
        img = load_image(imf)
        w, h = img.size
        top = np.random.randint(0, h-crop_size)
        left = np.random.randint(0, w-crop_size)
        crop = img.crop((left, top, left+crop_size, top+crop_size))
        # print(k_c)
        # print(crop.size)
        cropname = f"{Path(f_c).stem}_{top}_{left}" + "." + extension
        if output_dir:
            output_f = Path(output_dir) / cropname
            crop.save(str(output_f), mode="RGB")
        else:
            plt.imshow(crop)
            plt.title(cropname)
            plt.show()


if __name__=="__main__":
    args = parse_args()

    crop_dataset(args.input, args.n, args.output, args.size, args.extension)

    
    