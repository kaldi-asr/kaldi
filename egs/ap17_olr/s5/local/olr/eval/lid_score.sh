#!/bin/bash

# Copyright 2017 Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0

. ./path.sh
. ./cmd.sh

nj=20

dir=lid_net_output
mkdir -p $dir

lid_model=$1/final.raw
data=$2
sdata=$data/split$nj
split_data.sh $data $nj
data_base=`basename $data`
result=$1/lid_results_$data_base

for mdl in $lid_model; do
    echo "Nnet forwards ${data_base}."

    $cpu_cmd JOB=1:$nj $dir/log/lid_forward.${data_base}.JOB.log \
    nnet3-compute --use-gpu=no --extra-left-context=40 --frames-per-chunk=20 $mdl scp:$sdata/JOB/feats.scp ark,t:$dir/output.${data_base}.JOB.ark || exit 1

    if [ -f $dir/output.${data_base}.ark ]; then
      rm $dir/output.${data_base}.ark
    fi

    for job in `seq $nj`; do
      cat $dir/output.${data_base}.${job}.ark  >> $dir/output.${data_base}.ark
      rm $dir/output.${data_base}.${job}.ark
    done
    matrix-sum-rows ark:$dir/output.${data_base}.ark ark,t:$dir/output.${data_base}.ark.utt

    echo "Forwards ${data_base} done."
done

if [ ! -f $data/feats.len ]; then
  feat-to-len scp:$data/feats.scp ark,t:$data/feats.len || exit 1;
fi
python local/olr/eval/lid_utt_average.py $data $data_base || exit 1;
python local/olr/eval/lid_format_frame.py $data_base || exit 1;
python local/olr/eval/lid_format_utt.py $data_base || exit 1;

if [ -f $result ]; then
  rm $result
fi

for file in lid_score/{output.${data_base}.ark,output.${data_base}.ark.utt_average}; do
  if [[ $file == *utt* ]]; then
    echo "----  Utter level  ----" >> $result
  else
    echo "----  Frame level  ----" >> $result
  fi

  python local/olr/eval/Compute_Cavg.py $file >> $result
 
  mkdir -p lid_score/DET 
  python local/olr/eval/Prepare_Det.py  $file
  awk '{print $1," target"}'    lid_score/DET/target.txt > lid_score/trials.trl
  awk '{print $1," nontarget"}' lid_score/DET/nontarget.txt >> lid_score/trials.trl
  eer=$(compute-eer lid_score/trials.trl 2>/dev/null)
 
  printf '% 16s' 'EER% is:' >> $result
  printf '% 5.2f' $eer >> $result
  echo >> $result
  echo >> $result
done
rm -rf $dir lid_score

echo "Done the LID evaluation on $data_base, results in ${result}."
cat $result

echo "---- end ----"
exit 0;
