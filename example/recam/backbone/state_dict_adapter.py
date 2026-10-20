from collections import OrderedDict
from typing import Mapping

import torch


def filter_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    output_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if "img_emb" in key or "v_img" in key or "norm_k_img" in key or "k_img" in key:
            continue
        output_state_dict[key] = value
    return output_state_dict
