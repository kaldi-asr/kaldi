#!/bin/bash


[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
stats=true
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

data=$1
dir=$2

ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"


mkdir -p $dir/scoring_kaldi
if [ -f $data/reftext ]; then
  cat $data/reftext | $ref_filtering_cmd > $dir/scoring_kaldi/test_filt.txt || exit 1;
else
  echo "$0: No reference text to compute WER" 
fi

if [ $stage -le 0 ]; then

  mkdir -p $dir/scoring_kaldi/log
  # begin building hypothesis hyp.txt
  # in the same format as $data/reftext
  rm -rf tmpconcat
  awk '{a[$1]=a[$1]" "$5;}END{for(i in a)print i""a[i];}' \
    $dir/score_10/ctm_out > tmpconcat
  if [ -f $data/reftext ]; then
    awk -F" " '{print $1}' $data/reftext > tmpreforder
    rm -rf $dir/score_10/ctm_out.concat
    while read LINE; do
      grep "$LINE" "tmpconcat" >> "$dir/score_10/ctm_out.concat"
    done < "tmpreforder"
    rm -rf tmpconcat
    rm -rf tmpreforder
    $hyp_filtering_cmd $dir/score_10/ctm_out.concat > \
      $dir/scoring_kaldi/hyp.txt || exit 1;
    #end building hypothesis hyp.txt
    
    $cmd $dir/scoring_kaldi/log/score.hyp.log \
      cat $dir/scoring_kaldi/hyp.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.txt  ark:- ">&" $dir/wer || exit 1;

    cat $dir/wer
  else
    cat tmpconcat > "$dir/score_10/ctm_out.concat"
    rm -rf tmpconcat
    $hyp_filtering_cmd $dir/score_10/ctm_out.concat > \
      $dir/scoring_kaldi/hyp.txt || exit 1;
    exit 0;
    #end building hypothesis hyp.txt
  fi
fi

if [ $stage -le 1 ]; then
  if $stats; then
    mkdir -p $dir/scoring_kaldi/wer_details

    $cmd $dir/scoring_kaldi/log/stats1.log \
      cat $dir/scoring_kaldi/hyp.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring_kaldi/test_filt.txt ark:- ark,t:- \| \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring_kaldi/wer_details/per_utt \| \
      utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring_kaldi/wer_details/per_spk || exit 1;

    $cmd $dir/scoring_kaldi/log/stats2.log \
      cat $dir/scoring_kaldi/wer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring_kaldi/wer_details/ops || exit 1;

    $cmd $dir/scoring_kaldi/log/wer_bootci.log \
      compute-wer-bootci --mode=present \
         ark:$dir/scoring_kaldi/test_filt.txt ark:$dir/scoring_kaldi/hyp.txt \
         '>' $dir/scoring_kaldi/wer_details/wer_bootci || exit 1;
  fi
fi
