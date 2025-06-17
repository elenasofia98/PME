# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models
Private Memorization Editing (PME) is a model editing tecnique designed to protect LLMs against privacy attacks.
In fact, LLMs may memorize Personally Identifiable Information (PII) among huge amounts of uncontrolled data, and the right prompt might be sufficient to make the model leak that PII.
PME is an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. 
While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. 

In this repo, you can find the code used during the experiments for [Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](https://arxiv.org/abs/2506.10024)

### Replication
- Run pre-edit attacks in Attacks-PME: 01 notebooks will produce leaked PII from the selected model
- Run edit: baselines and PME (in the code, memoedit) can be runned from the notebooks in EasyEdit. DeMem baseline implementation can be found in DeMemorization-main
- Run post-edit evaluations: in Attacks-PME 02 notebooks are the post-edit attacks, 04 the notebooks for making tables and 09 allow to generate with pre and post edit models to quantify their similarity. LM Eval Harness on pre and post edit models is in lm-evaluation-harness

### Data
Please notes that we do not publicly share PII extracted from the Pile.
Feel free to reach out at ``elena.sofia.ruzzetti AT uniroma2.it`` to get the dataset for research purposes


### Resources
Original Repo used and modified:
- [EasyEdit](https://github.com/zjunlp/EasyEdit) for the edit via PME (added), MEMIT and Grace
- [nnsight](https://github.com/ndif-team/nnsight) for the intial study of contributions of FF blocks and Attention
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) for the post edit evaluation
- [DeMemorization](https://github.com/Alymostafa/DeMemorization/tree/main) for the DeMem baseline implementation
- [LM_PersonalInfoLeak](https://github.com/jeffhj/LM_PersonalInfoLeak) for the code and data on email adresses leaked

### Citations
Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models has been accepted as ACL 2025 Main Paper.
For more details about our method, the [preprint of our paper is available on arXiv](https://arxiv.org/abs/2506.10024)

```
@misc{ruzzetti2025privatememorizationeditingturning,
      title={Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models}, 
      author={Elena Sofia Ruzzetti and Giancarlo A. Xompero and Davide Venditti and Fabio Massimo Zanzotto},
      year={2025},
      eprint={2506.10024},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.10024}, 
}
```

More info coming soon
