<!-- git submodule add https://github.com/yzhangcs/parser.git 3rdparty/parser
git submodule add https://github.com/HillZhang1999/MuCGEC.git 3rdparty/mucgec
git submodule add https://github.com/mfelice/imeasure.git 3rdparty/imeasure

git submodule init
git submodule update -->


<div align="center">

# Improving Seq2Seq Grammatical Error Correction via Decoding Interventions
__Houquan Zhou__, Yumeng Liu, Zhenghua Li<sup title="Corresponding author" style="font-size:10px">✉️</sup>, Min Zhang, Bo Zhang, Chen Li, Ji Zhang, Fei Huang

</div>

<!-- A image -->
<div align="center">
<img src="cover.jpg" width="350" height="350" alt="cover" align=center />
<br>
<sup align=center>Note: This cover image is created by <a href="https://openai.com/dall-e-3">DALL·E 3</a></sup>
</div>
</div>

## TL;DR
This repo contains the code for our EMNLP 2023 Findings paper: [Improving Seq2Seq Grammatical Error Correction via Decoding Interventions](tbd).

We introduce a decoding intervention framework that uses *critics* to assess and guide token generation.
We evaluate two types of critics: **a pre-trained language model** and **a incremental target-side grammatical error detector**.
Experiments on English and Chinese data show our approach surpasses many existing methods and competes with SOTA models.

## Citation
```bib
@inproceedings{zhou-et-al-2023-improving,
  title     = {Improving Seq2Seq Grammatical Error Correction via Decoding Interventions},
  author    = {Zhou, Houquan  and
               Liu, Yumeng  and
               Li, Zhenghua  and
               Zhang, Min  and
               Zhang, Bo  and
               Li, Chen  and
               Zhang, Ji  and
               Huang, Fei},
  booktitle = {Findings of EMNLP},
  year      = {2023},
  address   = {Singapore}
}
```

## Setup

Clone this repo recursively:
```sh
git clone https://github.com/Jacob-Zhou/gecdi.git --recursive
# The newest version of parser is not compatible with the current code, 
# so we need to checkout to a previous version
cd 3rdparty/parser/ && git checkout 6dc927b
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

