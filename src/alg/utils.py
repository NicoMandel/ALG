# Utilities file
import os.path
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import numpy as np
import torch

def load_config(fpath : str) -> dict:
    with open(fpath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def compare_models(m1 : torch.nn.Module, m2 : torch.nn.Module, detail : bool = False) -> bool:
    """
        Function to compare the state dicts of two models to check if the parameters are equal.
        A check to see whether the parameters have been updated.
        If detail is true, will provide detailed check
    """
    if detail:
        return _compare_models_detail(m1, m2)
    else:
        return _compare_models(m1, m2)

def _compare_models(m1 : torch.nn.Module, m2 : torch.nn.Module, detail : bool = False) -> bool:
    """
        Function to compare the state dicts of two models to check if the parameters are equal.
        A check to see whether the parameters have been updated.
        If detail is true, will provide detailed check
    """
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def _compare_models_detail(m1 : torch.nn.Module, m2: torch.nn.Module) -> bool:
    models_differ = 0
    for key_item_1, key_item_2 in zip(m1.state_dict().items(), m2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception("Keys differ at: {} for model 1 and {} for model 2".format(key_item_1[0], key_item_2[0]))
    if models_differ == 0:
        print('Models match perfectly! :)')
        return True
    


def model_settings_from_args(args) -> dict:
    ms = vars(args)
    ms["optim"] = args.optim if args.optim else "adam",
    ms["lr"] = args.learning_rate if args.learning_rate else 1e-3
    ms["bs"] = args.batch_size if args.batch_size else 16
    ms["tune_fc_only"] = args.tune_fc_only if args.tune_fc_only else False
    ms["transfer"] = args.transfer if args.transfer else True
    ms["threshold"] = args.threshold if args.threshold else 0.5

    return ms

def get_subdirs(dirname : str) -> list[str]:
    return [os.path.join(dirname, name) for name in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, name))]

def copy_img_and_label(n : int | list, input_basedir : str, output_basedir : str, i_imgs : str = "input_images", i_labels : str = "mask_images", o_imgs : str = "images", o_labels : str = "labels", fext : str = ".tif"):
    input_imgdir = Path(input_basedir) / i_imgs
    input_labeldir = Path(input_basedir) / i_labels
    
    output_imgdir = Path(output_basedir) / o_imgs
    output_labeldir = Path(output_basedir) / o_labels

    if isinstance(n, int):
        img_list = list([x.stem for x in input_imgdir.glob("*" + fext)])
        img_ids = np.random.choice(img_list, n)
    else:
        img_ids = n
    
    for img_id in tqdm(img_ids):
        inp_img_f = input_imgdir / (img_id + fext)
        inp_lab_f = input_labeldir / (img_id + fext)
        outp_img_f = output_imgdir / (img_id + fext)
        outp_lab_f = output_labeldir / (img_id + fext)
        shutil.copy2(inp_img_f, outp_img_f)
        shutil.copy2(inp_lab_f, outp_lab_f)
    print("Copied {} images and associated label files from: {} to: {}".format(
        len(img_ids), input_basedir, output_basedir
    ))
