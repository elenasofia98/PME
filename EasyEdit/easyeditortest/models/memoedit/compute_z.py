from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .repr_tools import *
from ...util import nethook

from .memoedit_hparams import memoeditHyperParams



def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: memoeditHyperParams,
    layer: int,
    context_templates: List[str],
    return_original_contribution: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    
    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to(f"{hparams.device}")[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"{hparams.device}")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"{hparams.device}")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    layer_out = {hparams.layer_module_tmp.format(l): None for l in hparams.layers} ## will store the contribution of each layer to the final representation

    # Inserts new "delta" variable at the appropriate part of the computation
    # and store initial values layer-wise
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        nonlocal layer_out

        if cur_layer in layer_out and layer_out[cur_layer] is None:
            layer_out[cur_layer] = cur_out[0][0, lookup_idxs[0]].detach().clone()
            #print(cur_layer, layer_out[cur_layer].shape)
        
        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            
            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)


    prec_prob = None
    prob = None
    
    best_prob = None
    best_prob_delta = None
    
    prec_loss = None
    loss = None

    #best_delta = None
    
    epsilon = 1e-3
    it = 0
    it_no_gain = 0
    max_no_gain_iter = hparams.v_num_grad_steps

    to_rollback = False
    
    # Execute optimization
    while True:
        opt.zero_grad()
        all_layers = [
            hparams.layer_module_tmp.format(l) for l in hparams.layers
        ] + [
            hparams.layer_module_tmp.format(loss_layer), hparams.layer_module_tmp.format(layer)
        ]
        
        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=all_layers,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets

        output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        kl_factor = hparams.kl_factor
        if it_no_gain >= int(max_no_gain_iter / 2):
            kl_factor = hparams.kl_factor / 2
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
            
        
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"{it} - loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{prob}"
        )
        #### Obtained a good loss / prob 
        if loss < 1e-2:
            break
        if prob > 0.99 and loss < 3e-2:
            break

        ### safety exit
        if it > 500:
            print("Breaking, max iter, getting best prob results")
            print(f"Setting delta to best delta, prob = {best_prob}")
            print(delta == best_prob_delta)
            delta = best_prob_delta
            break
        
        if (prec_loss is not None and loss is not None and torch.absolute(loss - prec_loss) <= epsilon) or \
            (prec_prob is not None and prob is not None and abs(prob - prec_prob) <= epsilon) or \
            (prec_prob is not None and prob is not None and prob < prec_prob):
            it_no_gain += 1
        else:
            it_no_gain = 0
            
        if it_no_gain % 10 == 0 and it_no_gain>0:
            print(f"Iteration without sub. update = {it_no_gain}")
        
        if it_no_gain >= max_no_gain_iter: ## the loss is not moving
            print("Break, not optimizing for too many iterations")
            if prob < 0.10: # it has been too difficult to optimize
                print(f"Rollbacking changes --> {prob} too low")
                to_rollback = True
            break 

        
        if prec_prob is not None and prob is not None and prec_prob > prob and prec_prob - prob >= 0.07 and prec_prob > 0.93: 
            print(f"Break, not optimizing already good prob {prec_prob}")
            print(f"Setting delta to best delta, prob = {best_prob}")
            print(delta == best_prob_delta)
            delta = best_prob_delta
            break
        
        
        
        it += 1        
        
        prec_loss = loss
        prec_prob = prob


        if best_prob is None or best_prob < prob:
            best_prob = prob
            with torch.no_grad():
                best_prob_delta = delta.detach().clone()
        
        
        #OLDER
        #if it == hparams.v_num_grad_steps - 1:
        #    break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

        

    if prob < epsilon:
        to_rollback = True
    
    ### Compute contributions
    layer_out = torch.stack([layer_out[hparams.layer_module_tmp.format(l)] for l in hparams.layers], axis=1)
    layer_out = torch.cumsum(layer_out, axis=1)
    #print('layer_out.shape', layer_out.shape)
    
    print("Computing contributions!")
    #layer_out = torch.linalg.pinv(layer_out) @ target_init
    layer_out = layer_out.T @ target_init.unsqueeze(dim=1)
    #print("contributions", layer_out, layer_out.shape)
    layer_out = layer_out / target_init.norm()
    #print("contributions", layer_out, layer_out.shape)
    layer_out = layer_out / torch.sum(layer_out)
    #layer_out = layer_out / torch.sum(layer_out)
    #print("relative contribution", layer_out.shape)
    
    
    target = target_init + delta
    if not to_rollback:
        print(
            f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
        )
    else:
        print(
            f"Rollbacking init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
        )

    if not return_original_contribution:#return_original_output:
        return target, (delta.norm() == 0) or to_rollback
    else:
        return target, layer_out, (delta.norm() == 0) or to_rollback


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track=None,
    track_subupdates=False
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        #print("*"*80)
        #print('heeere')
        #print("*"*80)
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, track_subupdates=track_subupdates, **context_info, **word_repr_args
            )
        representations = get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, track_subupdates=track_subupdates, **context_info, **word_repr_args
        )
    #elif fact_token_strategy == "last":
    #    raise Exception("This is definitely bugged, fix it.")
    #    context_info = dict(
    #        contexts=[
    #            tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
    #        ],
    #        idxs=[000000],
    #    )
    #    if track == 'out' or track == 'in':
    #        return get_reprs_at_word_tokens(
    #            track=track, subtoken=subtoken, **context_info, **word_repr_args
    #        )
    #    l_input, l_output = get_reprs_at_idxs(
    #        track="both", **context_info, **word_repr_args
    #    )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    if track_subupdates:
        (l_input, l_output, mlp_subupdates) = representations
        return l_input.detach(), l_output.detach(), mlp_subupdates.detach()
    else:
        (l_input, l_output) = representations
        return l_input.detach(), l_output.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
