#!/usr/bin/env bash


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
  awk '{a[$1]=a[$1]" "$5;}END{for(i in a)print i""a[i];}' \
    $dir/score_10/ctm_out > tmpconcat
  if [ -f $data/reftext ]; then
    awk -F" " '{print $1}' $data/reftext > tmporder
    awk 'FNR==NR {x2[$1] = $0; next} $1 in x2 {print x2[$1]}' \
      tmpconcat tmporder > "$dir/score_10/ctm_out.concat"
    $hyp_filtering_cmd $dir/score_10/ctm_out.concat > \
      $dir/scoring_kaldi/hyp.txt || exit 1;
    # end building hypothesis hyp.txt

    $cmd $dir/scoring_kaldi/log/score.hyp.log \
      cat $dir/scoring_kaldi/hyp.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.txt  ark:- ">&" $dir/wer || exit 1;

    cat $dir/wer
  else
    cat tmpconcat > "$dir/score_10/ctm_out.concat"
    awk -F" " '{print $1}' $dir/score_10/ctm_out.concat > tmporder
    $hyp_filtering_cmd $dir/score_10/ctm_out.concat > \
      $dir/scoring_kaldi/hyp.txt || exit 1;
    #exit 0;
    #end building hypothesis hyp.txt

  fi
  
  # building hyp.segmentedXms.txt
  for dur in {700,800,900,1000}; do                                             
    dursec=`echo $dur' / 1000' | bc -l`                                         
    awk '{if ($4 < '$dursec') a[$1]=a[$1]" "$5; else a[$1]=a[$1]" "$5"\n"$1"";}END\
      {for(i in a)print i""a[i];}' $dir/score_10/ctm_out > tmpconcat          
    rm -rf $dir/score_10/ctm_out.concat.$dur                                    
    while read LINE; do                                                         
    grep "$LINE" "tmpconcat" >> "$dir/score_10/ctm_out.concat."$dur           
    done < "tmporder"                                                        
    
    $hyp_filtering_cmd $dir/score_10/ctm_out.concat.$dur > $dir/scoring_kaldi/hyp.segmented${dur}ms.txt || exit 1;                   
  done       
  rm -rf tmpconcat                                                            
  rm -rf tmporder 
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
