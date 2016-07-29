#!/bin/bash

. ./path.sh
. ./cmd.sh

stage=0

lang=data/lang
ali_dir=exp_clean/tri2b_ali

mdelta_stats_dir=tmp/mdelta_stats_dir
. utils/parse_options.sh

#pri file
mkdir -p $mdelta_stats_dir
if [ $stage -le 0 ]; then

  ali-to-phones --per-frame \
    $ali_dir/final.mdl \
    "ark:gunzip -c $ali_dir/ali.*.gz |" \
    ark,t:$mdelta_stats_dir/phones.tra
  
  echo "1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80" >$mdelta_stats_dir/dt.list
  
  perl utils/multi-stream/pm_utils/mtd/calc_pri_dt.pl \
    $mdelta_stats_dir/dt.list \
    $mdelta_stats_dir/phones.tra >$mdelta_stats_dir/pri
fi

#pdf_to_pseudo_phone.txt file
logdir=$mdelta_stats_dir/log; mkdir -p $logdir
if [ $stage -le 1 ]; then
  # Create pseudo_phones.txt
  utils/phone_post/create_pseudo_phones.py $lang/phones/roots.txt 2>$logdir/create_pseudo_phones.log >$mdelta_stats_dir/pseudo_phones.txt || exit 1;

  # Create pdf_to_pseudo_phones.txt
  model=$ali_dir/final.mdl
  show-transitions $lang/phones.txt "$model" 2>$logdir/show_transitions.log >$mdelta_stats_dir/show_transitions.txt || exit 1;

  utils/phone_post/show_transitions_to_sym2int.py $mdelta_stats_dir/show_transitions.txt $lang/phones/roots.txt | utils/sym2int.pl -f 2 $mdelta_stats_dir/pseudo_phones.txt | sort -n -k 1 | uniq > $mdelta_stats_dir/pdf_to_pseudo_phone.txt || exit 1;

  # Check if pdf-id's are uniq
  max=`tail -1 $mdelta_stats_dir/pdf_to_pseudo_phone.txt | awk '{print $1;}'`
  num_lines=`cat $mdelta_stats_dir/pdf_to_pseudo_phone.txt | wc -l`
  if [[ $max -ne $num_lines-1 ]]; then
    echo "pdf-id's are not uniq.";
    exit 1;
  fi
fi


