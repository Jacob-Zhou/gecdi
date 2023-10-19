#!/bin/bash
#SBATCH -p batch
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

# bash pred.sh devices=0,1,2,3 path=exp/transformer/model.1 config=configs/transformer.ini batch=10000
# bash pred.sh devices=0,1,2,3 path=exp/bart/model.1        config=configs/bart.ini        batch=5000

set -o nounset
set -o errexit
set -o pipefail

{
    . scripts/set_environment.sh
    args=( "$@" )
    for arg in "${args[@]}"; do
        eval "$arg"
    done

    echo "config:    ${config:=configs/bart.ini}"
    echo "path:      ${path:=$(mktemp -d)}"
    echo "suffix:    ${suffix:=}"
    echo "gec_path:  ${gec_path:=models/model.seed-1}"
    echo "lm_path:   ${lm_path:=gpt2}"
    echo "lm_alpha:  ${lm_alpha:=0.0}"
    echo "lm_beta:   ${lm_beta:=0}"
    echo "ged_path:  ${ged_path:=models/ged_model.seed-1}"
    echo "ged_alpha: ${ged_alpha:=0.0}"
    echo "ged_beta:  ${ged_beta:=0}"
    echo "devices:   ${devices:=0}"
    echo "batch:     ${batch:=1000}"
    echo "beam:      ${beam:=12}"
    echo "dataset:   ${dataset:=conll14.test}"
    echo "data:      ${data:=data/$dataset}"

    if [ "$(echo "$ged_alpha == 0" | bc -l)" -eq 1 ]; then
        if [ "$(echo "$lm_alpha == 0" | bc -l)" -eq 1 ]; then
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/baseline$suffix/$dataset.beam-"$beam".pred}"
        else
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/lm-$lm_path$suffix/$dataset.penalty-"$lm_alpha".$lm_beta.beam-"$beam".pred}"
        fi
    else
        if [ "$(echo "$lm_alpha == 0" | bc -l)" -eq 1 ]; then
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/ged$suffix/$dataset.penalty-"$ged_alpha".$ged_beta.beam-"$beam".pred}"
        else
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/ged-$lm_path$suffix/$dataset.penalty-"$ged_alpha".$ged_beta-"$lm_alpha".$lm_beta.beam-"$beam".pred}"
        fi
    fi

    mkdir -p "$(dirname "$pred")"

    python -u intervened_decode.py predict  \
        -d "$devices" -c "$config"  \
        --cache  \
        --path "$path/"  \
        --gec-path "$gec_path"  \
        --language "english"  \
        --lm-path=$lm_path  \
        --lm-alpha="$lm_alpha"  \
        --lm-beta="$lm_beta"  \
        --ged-path "$ged_path"  \
        --ged-alpha="$ged_alpha"  \
        --ged-beta="$ged_beta"  \
        --data "$data"  \
        --pred "$pred"  \
        --batch-size="$batch"  \
        --beam-size="$beam"  \
        --max-len=64  \
        --buckets 32  | tee "$pred".log

    {
        input="$pred.tmp.input"
        python scripts/predict_utils/para_to_input.py --input-file "$data" > "$input"
        CUDA_VISIBLE_DEVICES=-1 python eval_pred.py --hyp "$pred" -o "$pred".out -i "$input" -p "$gec_path" -m 62
        if [[ $data =~ "conll14" ]]; then
            python scripts/predict_utils/retokenize.py -s CoNLL "$pred".out > "$pred".retokenized.out &
            python2 tools/m2scorer/scripts/m2scorer.py -v "$pred".out  data/"$dataset".m2 | tee "$pred".m2scorer.log
            python2 tools/m2scorer/scripts/m2scorer.py -v "$pred".out  data/"$dataset".10x.m2 | tee "$pred".10x.m2scorer.log
        else
            python scripts/predict_utils/retokenize.py -s BEA "$pred".out > "$pred".retokenized.out &
            deactivate
            if [[ $dataset != "bea19.test" ]]; then
                . scripts/set_py36_environment.sh
                errant_parallel -orig "$input" -cor "$pred".out -out "$pred".m2
                errant_compare -v -hyp "$pred".m2 -ref data/"$dataset".m2 | tee "$pred".errant.log
                deactivate
            fi
        fi
        rm "$pred"
        rm "$pred".out
        rm "$pred".tmp.input
        rm -r "$path"
    } &
}