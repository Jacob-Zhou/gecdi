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

    echo "config:    ${config:=configs/bart.zh.ini}"
    echo "path:      ${path:=$(mktemp -d)}"
    echo "suffix:    ${suffix:=}"
    echo "gec_path:  ${gec_path:=models/model.zh.seed-1}"
    echo "lm_path:   ${lm_path:=uer/gpt2-chinese-cluecorpussmall}"
    echo "lm_alpha:  ${lm_alpha:=0.0}"
    echo "lm_beta:   ${lm_beta:=0}"
    echo "ged_path:  ${ged_path:=models/ged_model.zh.seed-1}"
    echo "ged_alpha: ${ged_alpha:=0.0}"
    echo "ged_beta:  ${ged_beta:=0}"
    echo "devices:   ${devices:=0}"
    echo "batch:     ${batch:=1000}"
    echo "beam:      ${beam:=12}"
    echo "dataset:   ${dataset:=mucgec.dev}"
    echo "data:      ${data:=data/cgec/$dataset.collapsed}"
    echo "gold:      ${gold:=data/cgec/$dataset.m2}"

    lm_name=$(sed -s "s\/\-\g" <<<  $lm_path)

    if [ "$(echo "$ged_alpha == 0" | bc -l)" -eq 1 ]; then
        if [ "$(echo "$lm_alpha == 0" | bc -l)" -eq 1 ]; then
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/baseline$suffix/$dataset.beam-"$beam".pred}"
        else
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/lm-$lm_name$suffix/$dataset.penalty-"$lm_alpha".$lm_beta.beam-"$beam".pred}"
        fi
    else
        if [ "$(echo "$lm_alpha == 0" | bc -l)" -eq 1 ]; then
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/ged$suffix/$dataset.penalty-"$ged_alpha".$ged_beta.beam-"$beam".pred}"
        else
            echo "pred:      ${pred:=$(dirname "$gec_path")/pred/$dataset/ged-$lm_name$suffix/$dataset.penalty-"$ged_alpha".$ged_beta-"$lm_alpha".$lm_beta.beam-"$beam".pred}"
        fi
    fi

    mkdir -p "$(dirname "$pred")"

    input="$pred.tmp.input.ori"
    python scripts/predict_utils/para_to_input.py --input-file "$data" > "$input"
    if [[ $dataset =~ "mucgec" ]]; then
        CUDA_VISIBLE_DEVICES=-1 python scripts/predict_utils/split_discourse.py --style MuCGEC --input-file $input --output-file "$pred".tmp --path "$gec_path" -m 126
    else
        CUDA_VISIBLE_DEVICES=-1 python scripts/predict_utils/split_discourse.py --style NLPCC --input-file $input --output-file "$pred".tmp --path "$gec_path" -m 126
    fi
    # python scripts/predict_utils/input_to_pid.py "$pred".tmp.input > "$pred".tmp.input.pid
    # data="$pred".tmp.input.pid
    data="$pred".tmp

    python -u intervened_decode.py predict  \
        -d "$devices" -c "$config"  \
        --cache  \
        --path "$path/"  \
        --gec-path "$gec_path"  \
        --language "chinese"  \
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
        --max-len=128  \
        --buckets 32  | tee "$pred".log

    {
        CUDA_VISIBLE_DEVICES=-1 python eval_pred_zh.py --hyp "$pred" -o "$pred".split -i "$pred".tmp.input -p "$gec_path" -m 126
        python scripts/predict_utils/compose_discourse_input.py --input-file $pred.split  --index-file "$pred".tmp.index --output-file "$pred".out
        if [[ $dataset =~ "nlpcc18" ]]; then
            python scripts/predict_utils/retokenize_zh.py -s NLPCC "$pred".out > "$pred".retokenized.out
            python2 tools/m2scorer/scripts/m2scorer.py -v "$pred".retokenized.out "$gold" | tee "$pred".m2scorer.log
        else
            # preprocess compose sentences
            paste "$input" "$pred".out | awk '{print NR"\t"$p}' > "$pred".para
            workdir=`pwd`
            cd tools/ChERRANT
            python parallel_to_m2.py -f "$workdir"/"$pred".para -o "$workdir"/"$pred".m2 -g char
            python compare_m2_for_evaluation.py -v -hyp "$workdir"/"$pred".m2 -ref "$workdir"/"$gold" | tee "$workdir"/"$pred".cherrant.log
            cd $workdir
        fi
        rm "$pred".split
        rm "$pred".tmp*
    } &
}