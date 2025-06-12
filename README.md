Code used during the experiments for "Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models"

### Replication
- Run pre-edit attacks in Attacks-PME: 01 notebooks will produce leaked PII from the selected model
- Run edit: baselines and PME (in the code, memoedit) can be runned from the notebooks in EasyEdit. DeMem baseline implementation can be found in DeMemorization-main
- Run post-edit evaluations: in Attacks-PME 02 notebooks are the post-edit attacks, 04 the notebooks for making tables and 09 allow to generate with pre and post edit models to quantify their similarity. LM Eval Harness on pre and post edit models is in lm-evaluation-harness


### Resources
Original Repo used and modified:
- [EasyEdit](https://github.com/zjunlp/EasyEdit) for the edit via PME (added), MEMIT and Grace
- [nnsight](https://github.com/ndif-team/nnsight) for the intial study of contributions of FF blocks and Attention
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) for the post edit evaluation
- [DeMemorization](https://github.com/Alymostafa/DeMemorization/tree/main) for the DeMem baseline implementation
- [LM_PersonalInfoLeak](https://github.com/jeffhj/LM_PersonalInfoLeak) for the code and data on email adresses leaked

