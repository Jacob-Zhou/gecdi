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
<sup align=center>Note: This cover image is created by <a href="https://openai.com/dall-e-3">DALL·E 3</sup>
</div>
</div>

## TL;DR
This repo contains the code for our EMNLP 2021 Findings paper: [Improving Seq2Seq Grammatical Error Correction via Decoding Interventions](tbd).

We focuses on improving the Sequence-to-Sequence (Seq2Seq) method for grammatical error correction (GEC).
While Seq2Seq is promising for GEC, it faces challenges with limited and noisy training data and lacks token correctness awareness during decoding.
We introduce a decoding intervention framework that uses an external critic to assess and guide token generation.
We evaluate two types of critics: a pre-trained language model and a incremental target-side grammatical error detector.
Experiments on English and Chinese data show our approach surpasses many existing methods and competes with top-performing techniques.

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


## Run

English experiments:
```sh
# Baseline (vanilla decoding)
bash pred.sh devices=0 dataset=bea19.dev

# w/ LM-critic
bash pred.sh devices=0 lm_alpha=0.8  lm_beta=10 dataset=bea19.dev

# w/ GED-critic
bash pred.sh devices=0 ged_alpha=0.8 ged_beta=1 dataset=bea19.dev

# w/ both LM-critic and GED-critic
bash pred.sh devices=0 lm_alpha=0.8 lm_beta=10 ged_alpha=0.8 ged_beta=1 dataset=bea19.dev
```

Chinese experiments:
```sh
# Baseline (vanilla decoding)
bash pred.sh devices=0 dataset=mucgec.dev

# w/ LM-critic
bash pred.sh devices=0 lm_alpha=0.3  lm_beta=0.1 dataset=mucgec.dev

# w/ GED-critic
bash pred.sh devices=0 ged_alpha=0.6 ged_beta=10 dataset=mucgec.dev

# w/ both LM-critic and GED-critic
bash pred.sh devices=0 lm_alpha=0.3 lm_beta=0.1 ged_alpha=0.6 ged_beta=10 dataset=mucgec.dev
```

