#!/bin/bash
#
# Copyright 2020 Srikanth Madikeri (Idiap Research Institute)
# Apache 2.0
#
# This script combines egs folder generated with chain2 recipes to prepare a single egs folder
# for multilingual training

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e

# Begin configuration section
cmd=
block_size=256
stage=0
frames_per_job=1500000  
left_context=13
right_context=9
# TODO: add lang2weight support
lang2weight=            # array of weights one per input languge to scale example's output
                        # w.r.t its input language during training.
lang_list=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

[[ -f local.conf ]] && . local.conf

if [ $# -lt 3 ]; then
  cat <<EOF
  This script generates examples for multilingual LF-MMI training.
  The input egs directories are generated with chain2 get_egs scripts.

  Usage: $0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>
   e.g.: $0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs

  Options:
      --cmd (utils/run.pl|utils/queue.pl <queue opts>)  # how to run jobs.
EOF
  exit 1;
fi

num_langs=$1
if [ $# != $[$num_langs+2] ]; then
  echo "$0: num of input example dirs provided is not compatible with num_langs $num_langs."
  echo "Usage:$0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>"
  echo "Usage:$0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs"
  exit 1;
fi
megs_dir=${@: -1} # multilingual directory
mkdir -p $megs_dir
shift 1
args=("$@")

required="info.txt train.scp train_subset.scp heldout_subset.scp"
train_scp_list=
train_diagnostic_scp_list=
valid_diagnostic_scp_list=
combine_scp_list=

# we don't copy lang because there wont be a single lang
check_params="feat_dim left_context right_context left_context_initial right_context_final ivector_dim" 
ivec_dim=`fgrep ivector_dim ${args[0]}/info.txt | awk '{print $2}'`
# if [ $ivec_dim -ne 0 ];then check_params="$check_params final.ie.id"; fi

echo "dir_type randomized_chain_egs" > $megs_dir/info.txt
for param in $check_params frames_per_chunk; do
    awk "/^$param/" ${args[0]}/info.txt
    
done >> $megs_dir/info.txt
# the arguments to grep make sure we only grep the line that starts with excatly the word lang and we take the first such line
#lang_list=$(for i in `seq 0 $num_langs`; do awk '/^lang/' ${args[0]}/info.txt | awk '{print $2}'; done)
echo "langs ${lang_list[@]}" >> $megs_dir/info.txt

tot_num_archives=0
tot_num_scps=0
for lang in $(seq 0 $[$num_langs-1]);do
  multi_egs_dir[$lang]=${args[$lang]}
  for f in $required; do
    if [ ! -f ${multi_egs_dir[$lang]}/$f ]; then
      echo "$0: no such file ${multi_egs_dir[$lang]}/$f." && exit 1;
    fi
  done
  num_chunks=$(fgrep num_chunks ${multi_egs_dir[$lang]}/info.txt | awk '{print $2}')
  curr_frames_per_chunk_avg=`awk '/^frames_per_chunk_avg/  {print $2;}' ${multi_egs_dir[$lang]}/info.txt`
  tot_num_archives=$[tot_num_archives+((num_chunks*curr_frames_per_chunk_avg)/frames_per_job+1)]
  tot_num_scps=$[tot_num_scps+num_scps]
  train_diagnostic_scp_list="$train_diagnostic_scp_list ${args[$lang]}/train_subset.scp"
  valid_diagnostic_scp_list="$valid_diagnostic_scp_list ${args[$lang]}/valid_subset.scp"
  for f in $check_params; do
    if [ `grep -c "^$f" ${multi_egs_dir[$lang]}/info.txt` -ge 1 ]; then
      f1=$(fgrep -m 1 $f $megs_dir/info.txt | awk '{print $2}')
      f2=$(fgrep -m 1 $f ${multi_egs_dir[$lang]}/info.txt | awk '{print $2}')
      if [ "$f1" != "$f2" ]  ; then
        echo "$0: mismatch for $f in $megs_dir vs. ${multi_egs_dir[$lang]}($f1 vs. $f2)."
        exit 1;
      fi
    else
      echo "$0: parameter $f does not exist in $megs_dir or ${multi_egs_dir[$lang]}/$f ."
    fi
  done
done
num_scp_files=$tot_num_archives
echo "num_scp_files $num_scp_files" >> $megs_dir/info.txt
sed_cmd=
for lang in $(seq 0 $[$num_langs-1]);do
    lang_name=${lang_list[$lang]}
    weight=`echo $lang2weight | tr ',' ' ' | cut -d ' ' -f$[$lang+1]`
    sed_cmd="$sed_cmd s/.*lang=${lang_name}.*/$weight/;"
done

dir=$megs_dir/
if [ $stage -le 0 ]; then
    echo "$0: Creating $num_scp_files scp files."
    for lang in $(seq 0 $[$num_langs-1]);do
        lang_name=${lang_list[$lang]}
        [ ! -d $dir/temp_${lang_name}/ ] && mkdir $dir/temp_${lang_name}/
        # randomize, append language name as a query and split input scp into $num_blocks blocks
        utils/shuffle_list.pl ${args[$lang]}/train.scp | \
            awk -v lang_name="$lang_name" \
                '{if ($1 !~ /?/){$1=$1"?lang=" lang_name; print;} else {$1=$1"&lang=" lang_name; print;}}' > $dir/temp_${lang_name}/train.shuffled.scp 
            utils/split_scp.pl $dir/temp_${lang_name}/train.shuffled.scp \
                $(for i in $(seq $num_scp_files); do echo $dir/temp_${lang_name}/train.$i.scp; done) || exit 1
        # split each block into sub-blocks
        for i in `seq $num_scp_files`; do
            utils/split_scp.pl <(utils/shuffle_list.pl $dir/temp_${lang_name}/train.$i.scp) \
                $(for j in $(seq $num_scp_files); do echo $dir/temp_${lang_name}/train.$i.$j.scp; done)
        done
    done

    for j in `seq $num_scp_files`; do
        input_list=$(for lang in $(seq 0 $[$num_langs-1]);do lang_name=${lang_list[$lang]}; echo $dir/temp_${lang_name}/train.*.$j.scp; done)
        # the shuffling is probably not required because we will do it once again before
        # merging examples
        cat $input_list | utils/shuffle_list.pl > $dir/train.$j.scp
        sed "$sed_cmd" < <(awk '{print $1}' $dir/train.$j.scp) > $dir/train.weight.$j.ark.col2
        paste -d ' ' <(awk '{print $1}' $dir/train.$j.scp) $dir/train.weight.$j.ark.col2 > $dir/train.weight.$j.ark
        rm $dir/train.weight.$j.ark.col2
    done
fi

if [ $stage -le 1 ]; then
    for subset_file  in train_subset heldout_subset; do
        for lang in $(seq 0 $[$num_langs-1]);do
            lang_name=${lang_list[$lang]}
            cat ${args[$lang]}/${subset_file}.scp  | \
            awk -v lang_name="$lang_name" \
                '{if ($1 !~ /?/){$1=$1"?lang=" lang_name; print;} else {$1=$1"&lang=" lang_name; print;}}' 
        done > $dir/${subset_file}.scp
        sed "$sed_cmd" < <(awk '{print $1}' $dir/${subset_file}.scp) > $dir/${subset_file}.weight.ark.col2
        paste -d ' ' <(awk '{print $1}' $dir/${subset_file}.scp) $dir/${subset_file}.weight.ark.col2 > $dir/${subset_file}.weight.ark
        rm $dir/${subset_file}.weight.ark.col2
    done
fi

if [ $stage -le 2 ]; then
    echo "$0: Clean up"
    for lang in $(seq 0 $[$num_langs-1]);do
        lang_name=${lang_list[$lang]}
        rm -r $dir/temp_${lang_name}/
    done
fi

echo "$0: Finished preparing multilingual training example."
