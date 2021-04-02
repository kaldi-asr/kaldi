#! /bin/bash 

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Sorts and reports results in results/results.txt
# for all models in exp. Expects decode directories 
# to be named as exp/<model>/decode* or exp/chain/tdnn*/decode*
# Should be run from egs/cmu_cslu_kids.

res=${1:-"results/results.txt"}
exp=exp
mkdir -p results
rm -f $res

echo "Sorting results in: "
echo "# ---------- GMM-HMM Models ----------" >> $res
for mdl in $exp/mono* $exp/tri*; do
    echo "  $mdl"
    if [ -d $mdl ];then
        for dec in $mdl/decode*;do
            echo "    $dec"
            if [ -d $dec ];then
                grep WER $dec/wer* | \
                    sort -k2 -n > $dec/WERs
                head -n 1 $dec/WERs >> $res
            fi
        done
    fi
done

echo "# ---------- DNN-HMM Models ----------" >> $res
# DNN results
for mdl in $exp/chain/tdnn*; do
    echo "  $mdl"
    for dec in $mdl/decode*; do
        if [ -d $dec ]; then
            echo "    $dec"
            grep WER $dec/wer* | \
                sort -k2 -n > $dec/WERs
            head -n 1 $dec/WERs >> $res
        fi
    done
done

sed -i "s/:/    /g" $res 
