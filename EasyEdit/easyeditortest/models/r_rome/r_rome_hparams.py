from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class R_ROMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: str
    model_name: str
    stats_dir: str

    max_length: int = 40
    model_parallel: bool = False
    fp16: bool = False

    # compute key vector from prompt only
    enable_prompt_keys: bool = False
    # compute key vector as avg over random prefixes + prompt
    enable_random_prefix_keys: bool = True
    # Original ROME implementation overrides other options, uses both computations in the update equation
    original_implementation: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config["alg_name"] == "R-ROME") or print(
            f'R_ROMEHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} '
        )
        return cls(**config)