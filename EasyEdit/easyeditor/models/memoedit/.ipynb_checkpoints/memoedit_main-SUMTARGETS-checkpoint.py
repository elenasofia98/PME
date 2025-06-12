import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memoedit_hparams import memoeditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def get_vram():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    print(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')


def apply_memoedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: memoeditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memoedit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(f"{hparams.device}"), val_mat.to(f"{hparams.device}")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


TO_CHECK = True

def execute_memoedit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: memoeditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMOedit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    print("*"*80)
    print("NEW memoedit TEST!")
    print("*"*80)
    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    for request in requests[:10]:
        print(
            f"MEMOedit request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
   
    for request in requests:
        # Compute k/v pair
        cur_z = compute_z(
            model,
            tok,
            request,
            hparams,
            z_layer,
            context_templates,
        )

        z_list.append(cur_z)

    zs = torch.stack(z_list, dim=1)
    permuted_zs = zs.T.unsqueeze(dim=2)#.unsqueeze(dim=2) # (2,1,10)
    print("permuted_zs.shape", permuted_zs.shape)
    
    # Insert
    for i, layer in enumerate(hparams.layers):
        print("*"*40)
        print(f"\n\nLAYER {layer}\n")
        print("*"*40)

        
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        ################ Compute values ###############
        out = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out',
            track_subupdates=True
        )
        cur_zs, mlp_subupdates = out[0], out[1]
        
        #print('zs.shape', zs.shape) # zs.shape torch.Size([4096, 2])
        #print('cur_zs.shape', cur_zs.shape) # cur_zs.shape torch.Size([4096, 2])
        #print('mlp_subupdates', mlp_subupdates.shape) # mlp_subupdates torch.Size([4096, 16385, 2])

        #permuted_zs = torch.unsqueeze(zs, dim=1)
        #permuted_zs = torch.permute(permuted_zs, (2, 1, 0))
        #permuted_zs = zs.T.unsqueeze(dim=1)
        #permuted_mlp_subupdates = torch.permute(mlp_subupdates, (2, 0, 1))
        #permuted_mlp_subupdates.shape
        
        #sims = torch.bmm(permuted_zs, permuted_mlp_subupdates)
        #sims.shape #batch, 1, hidden_dim
        #print("sims", sims)
        #rescaled_subupdates = permuted_mlp_subupdates * sims
        #zs_at_layer = torch.sum(rescaled_subupdates, axis=-1).T
        #print(zs_at_layer.shape)

        TO_CHECK= False

        ############### COMPUTE COSINE SIMILARITY ##################
        #print("cur_zs.shape", cur_zs.shape)
        #permuted_zs = cur_zs.unsqueeze(dim=2)#.unsqueeze(dim=2) # (2,1,10)
        #print("permuted_zs.shape", permuted_zs.shape)
        
        print("mlp_subupdates", mlp_subupdates.shape)
        mlp_subupdates = mlp_subupdates.squeeze(dim=-3).squeeze(dim=0)
        mlp_subupdates = mlp_subupdates.permute(0,2,1)
        print("mlp_subupdates after squeeze amd reshape", mlp_subupdates.shape)

        
        (bs_size, hidden_dim, d) = mlp_subupdates.shape
        normalize=True
        
        if not normalize:
            sims = torch.bmm(mlp_subupdates, permuted_zs)

            if TO_CHECK:
                tot = 0
                correct = 0
                for bs in range(bs_size):
                    for seq in range(hidden_dim):
                        correct += (round(sims[bs][seq][0].item(), 3) == round(torch.dot(mlp_subupdates[bs][seq], zs[:, bs]).item(), 3))
                        tot +=1
                print(f"sims {correct}/{tot}")
        else:
            sims = torch.bmm(F.normalize(mlp_subupdates, p=2, dim=-2), 
                             F.normalize(permuted_zs, dim=1))
            
        print('sims.shape, normalized', sims.shape, normalize)
        (s, m) = torch.std_mean(sims, dim=1)
        print('average sim:', s, "+-", m)
        ############### COMPUTE COSINE SIMILARITY ##################



        ################## COMPUTE TARGETS #########################
        targets = mlp_subupdates*sims #scaled subupdates
        print("scaled_subupdates.shape", targets.shape)
        
        if TO_CHECK:
            tot = 0
            correct = 0
        
            for bs in range(bs_size):
                for seq in range(hidden_dim):
                    correct += sum((targets[bs][seq] == (mlp_subupdates[bs][seq] * sims[bs][seq][0].item()))) == d
                    tot += 1
            print(f"scaling {correct}/{tot}")

        targets = torch.sum(targets, dim=1).T #zs - cur_zs # zs_at_layer
        #print("targets", targets)
        print("targets.shape", targets.shape)
        print("targets.norm", torch.linalg.norm(targets, dim=0))

        # Empy cache
        mlp_subupdates.to('cpu')
        sims.to('cpu')

        del mlp_subupdates
        del sims
        torch.cuda.empty_cache()
        get_vram()
        

        
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        
        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )
        resid = targets #/ (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T


        # Adjust update matrix .shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"{hparams.device}")) if inv else COV_CACHE[key].to(f"{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]]
    return CONTEXT_TEMPLATES_CACHE
