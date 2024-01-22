<div align="center">

# Improving Seq2Seq Grammatical Error Correction via Decoding Interventions
__Houquan Zhou__, Yumeng Liu, Zhenghua Li<sup title="Corresponding author" style="font-size:10px">✉️</sup>, Min Zhang, Bo Zhang, Chen Li, Ji Zhang, Fei Huang

</div>

<!-- A image -->
<div align="center">
<img src="assets/cover-2.jpg" width="350" height="350" alt="cover" align=center />
<br>
<sup align=center>Note: This cover image is created by <a href="https://openai.com/dall-e-3">DALL·E 3</a></sup>
</div>
</div>

## TL;DR
This repo contains the code for our EMNLP 2023 Findings paper: [Improving Seq2Seq Grammatical Error Correction via Decoding Interventions](https://arxiv.org/abs/2310.14534).

We introduce a decoding intervention framework that uses *critics* to assess and guide token generation.
We evaluate two types of critics: **a pre-trained language model** and **an incremental target-side grammatical error detector**.
Experiments on English and Chinese data show our approach surpasses many existing methods and competes with SOTA models.

## Citation
```bib
@inproceedings{zhou-etal-2023-improving-seq2seq,
    title = "Improving {S}eq2{S}eq Grammatical Error Correction via Decoding Interventions",
    author = "Zhou, Houquan  and
      Liu, Yumeng  and
      Li, Zhenghua  and
      Zhang, Min  and
      Zhang, Bo  and
      Li, Chen  and
      Zhang, Ji  and
      Huang, Fei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.495",
    pages = "7393--7405",
}
```

## Setup

Clone this repo recursively:
```sh
git clone https://github.com/Jacob-Zhou/gecdi.git --recursive

# The newest version of parser is not compatible with the current code, 
# so we need to checkout to a previous version
cd gecdi/3rdparty/parser/ && git checkout 6dc927b && cd -
```

Then you can use following commands to create an environment and install the dependencies:
```sh
. scripts/set_environment.sh

# for Errant (v2.0.0) evaluation a python 3.6 environment is required
# make sure your system has python 3.6 installed, then run:
. scripts/set_py36_environment.sh
```


You can follow this [repo](https://github.com/HillZhang1999/SynGEC) to obtain the 3-stage train/dev/test data for training a English GEC model.
The multilingual datasets are available [here](https://github.com/google-research-datasets/clang8).

Before running, you are required to preprocess each sentence pair into the format of 
```txt
S   [src]
T   [tgt]

S   [src]
T   [tgt]
```
Where `[src]` and `[tgt]` are the source and target sentences, respectively.
A `\t` is used to separate the prefix `S` or `T` and the sentence.
Each sentence pair is separated by a blank line.
See [`data/toy.train`](data/toy.train) for examples.


## Download Trained Models
The trained models are avaliable in HuggingFace model hub.
You can download them by running:
```sh
# If you have not installed git-lfs, please install it first
# The installation guide can be found here: https://git-lfs.github.com/
# Most of the installation guide requires root permission.
# However, you can install it locally using conda:
# https://anaconda.org/anaconda/git-lfs

# Create directory for storing the trained models
mkdir -p models
cd models

# Download the trained models
# First, clone the small files
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HQZhou/bart-large-gec
# Then use git-lfs to download the large files
cd bart-large-gec
git lfs pull

# Return to the models directory
cd -

# The download process is the same for the GED model
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HQZhou/bart-large-ged
cd bart-large-ged
git lfs pull

# The download process is the same for the Chinese models
# Just change the GEC url to https://huggingface.co/HQZhou/bart-large-chinese-gec
# and the GED url to https://huggingface.co/HQZhou/bart-large-chinese-ged
```

The models can also download by using the `huggingface-cli`:
```sh

# First make sure that you have installed `huggingface_hub` package
# You can install it following the guide here: https://huggingface.co/docs/huggingface_hub/installation
huggingface-cli download HQZhou/bart-large-gec --local-dir-use-symlinks False --local-dir models/bart-large-gec
huggingface-cli download HQZhou/bart-large-ged --local-dir-use-symlinks False --local-dir models/bart-large-ged
```

## Run

English experiments:
```sh
# Baseline (vanilla decoding)
bash pred.sh  \
    devices=0  \
    gec_path=models/bart-large-gec/model  \
    dataset=bea19.dev

# w/ LM-critic
bash pred.sh  \
    devices=0  \
    gec_path=models/bart-large-gec/model  \
    lm_alpha=0.8 lm_beta=10  \
    dataset=bea19.dev

# w/ GED-critic
bash pred.sh  \
    devices=0  \
    gec_path=models/bart-large-gec/model  \
    ged_path=models/bart-large-ged/model  \
    ged_alpha=0.8 ged_beta=1  \
    batch=500  \
    dataset=bea19.dev

# w/ both LM-critic and GED-critic
bash pred.sh  \
    devices=0  \
    gec_path=models/bart-large-gec/model  \
    ged_path=models/bart-large-ged/model  \
    lm_alpha=0.8 lm_beta=10  \
    ged_alpha=0.8 ged_beta=1  \
    batch=250  \
    dataset=bea19.dev
```

Chinese experiments:
```sh
# Baseline (vanilla decoding)
bash pred.sh  \
    devices=0  \
    dataset=mucgec.dev

# w/ LM-critic
bash pred.sh  \
    devices=0  \
    lm_alpha=0.3  \
    lm_beta=0.1  \
    dataset=mucgec.dev

# w/ GED-critic
bash pred.sh  \
    devices=0  \
    ged_alpha=0.6 ged_beta=10  \
    dataset=mucgec.dev

# w/ both LM-critic and GED-critic
bash pred.sh  \
    devices=0  \
    lm_alpha=0.3 lm_beta=0.1  \
    ged_alpha=0.6 ged_beta=10  \
    dataset=mucgec.dev
```

Run target-side GED only:
```sh
bash pred_ged.sh  \
    devices=0  \
    path=models/bart-large-ged/model  \
    data=<path to the parallel data to be detected>  \
    pred=<path to the output file>

# the output file is in the format of jsonl as follows:
# {
#     "src_text": "I implicated my class from winning the champion .",
#     "tgt_text": "I implicated my class in winning the champion .",
#     "tgt_subword": ["ĠI", "Ġimplicated", "Ġmy", "Ġclass", "Ġin", "Ġwinning", "Ġthe", "Ġchampion", "Ġ."],
#     "error": [[1, 2, "SUB"], [4, 5, "SUB"]]
# }

# the error field is a list of error spans, each span is represented as a list of three elements:
# [start of subword span, end of subword span, error type]
# error type can be one of the following:
# `RED`: redundant
# `SUB`: substitution
# `MISS-L`: there are missing tokens on the left side of the span
```

## Recommended Hyperparameters
We search the coefficient $\alpha$ and $\beta$ on the development set.

The optimal coefficients are varied across different datasets.

Hyperparameters for LM-critic:
| Dataset | $\alpha$ | $\beta$ |
|:-------:|:--------:|:-------:|
| CoNLL-14 | 0.8 | 10.0 |
| BEA-19 | 0.8 | 10.0 |
| GMEG-Wiki | 1.0 | 10.0 |
| MuCGEC | 0.3 | 0.1 |

Hyperparameters for GED-critic:
| Dataset | $\alpha$ | $\beta$ |
|:-------:|:--------:|:-------:|
| CoNLL-14 | 0.8 | 1.0 |
| BEA-19 | 0.8 | 1.0 |
| GMEG-Wiki | 0.9 | 1.0 |
| MuCGEC | 0.6 | 10.0 |

## Typo

- Appendix B.2 (STAGE 3): We further fine-tune the model on the W&I + LOCNESS **test** set only. $\rightarrow$ We further fine-tune the model on the W&I + LOCNESS **training** set only. (We sincerely apologize for this typo and thank @GMago123 for pointing it out in the [issue#4](https://github.com/Jacob-Zhou/gecdi/issues/4))
