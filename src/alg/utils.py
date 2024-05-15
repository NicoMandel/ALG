# Utilities file
import yaml
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