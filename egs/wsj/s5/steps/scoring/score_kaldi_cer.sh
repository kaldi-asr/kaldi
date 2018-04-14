#!/bin/bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0

# This script computes the CER (Character Error Rate) as opposed to the script
# local/score_kaldi.sh (which computes WER i.e. Word Error Rate).
# if you need to compute both the WER and CER, you can use the stage parameters
# i.e. write your own local/score.sh that will contain
# 
# steps/scoring/score_kaldi_wer.sh "$@"
# steps/scoring/score_kaldi_cer.sh --stage 2 "$@"
#
# NOTE it would work without the --stage 2, but this way its more effective
# as the lattice decoding won't be run twice.


[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
decode_mbr=false
stats=true
beam=6
stage=0
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
iter=final
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"


if $decode_mbr ; then
  echo "$0: scoring with MBR, word insertion penalty=$word_ins_penalty"
else
  echo "$0: scoring with word insertion penalty=$word_ins_penalty"
fi


mkdir -p $dir/scoring_kaldi
cat $data/text | $ref_filtering_cmd > $dir/scoring_kaldi/test_filt.txt || exit 1;
if [ $stage -le 0 ]; then

  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    mkdir -p $dir/scoring_kaldi/penalty_$wip/log

    if $decode_mbr ; then
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/best_path.LMWT.log \
        acwt=\`perl -e \"print 1.0/LMWT\"\`\; \
        lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
        lattice-prune --beam=$beam ark:- ark:- \| \
        lattice-mbr-decode  --word-symbol-table=$symtab \
        ark:- ark,t:- \| \
        utils/int2sym.pl -f 2- $symtab \| \
        $hyp_filtering_cmd '>' $dir/scoring_kaldi/penalty_$wip/LMWT.txt || exit 1;

    else
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/best_path.LMWT.log \
        lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
        lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
        utils/int2sym.pl -f 2- $symtab \| \
        $hyp_filtering_cmd '>' $dir/scoring_kaldi/penalty_$wip/LMWT.txt || exit 1;
    fi

    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/score.LMWT.log \
      cat $dir/scoring_kaldi/penalty_$wip/LMWT.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT_$wip || exit 1;

  done
fi


# the stage 2 is intentional, to allow nice coexistence with score_kaldi.sh
# in cases user would be combining calls to these two scripts as shown in
# the example at the top of the file. Otherwise we or he/she would have to
# filter the script parameters instead of simple forwarding.
if [ $stage -le 2 ] ; then
  files=($dir/scoring_kaldi/test_filt.txt)
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      files+=($dir/scoring_kaldi/penalty_${wip}/${lmwt}.txt)
    done
  done

  for f in "${files[@]}" ; do
    fout=${f%.txt}.chars.txt
    if [ -x local/character_tokenizer ]; then
      cat $f |  local/character_tokenizer > $fout
    else
      cat $f |  perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $fout
    fi
  done

  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring_kaldi/penalty_$wip/log/score.cer.LMWT.log \
      cat $dir/scoring_kaldi/penalty_$wip/LMWT.chars.txt \| \
      compute-wer --text --mode=present \
      ark:$dir/scoring_kaldi/test_filt.chars.txt  ark,p:- ">&" $dir/cer_LMWT_$wip || exit 1;
  done
fi

if [ $stage -le 3 ] ; then
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      # adding /dev/null to the command list below forces grep to output the filename
      grep WER $dir/cer_${lmwt}_${wip} /dev/null
    done
  done | utils/best_wer.sh  >& $dir/scoring_kaldi/best_cer || exit 1

  best_cer_file=$(awk '{print $NF}' $dir/scoring_kaldi/best_cer)
  best_wip=$(echo $best_cer_file | awk -F_ '{print $NF}')
  best_lmwt=$(echo $best_cer_file | awk -F_ '{N=NF-1; print $N}')

  if [ -z "$best_lmwt" ]; then
    echo "$0: we could not get the details of the best CER from the file $dir/cer_*.  Probably something went wrong."
    exit 1;
  fi

  if $stats; then
    mkdir -p $dir/scoring_kaldi/cer_details
    echo $best_lmwt > $dir/scoring_kaldi/cer_details/lmwt # record best language model weight
    echo $best_wip > $dir/scoring_kaldi/cer_details/wip # record best word insertion penalty

    $cmd $dir/scoring_kaldi/log/stats1.cer.log \
      cat $dir/scoring_kaldi/penalty_$best_wip/${best_lmwt}.chars.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring_kaldi/test_filt.chars.txt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring_kaldi/cer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring_kaldi/cer_details/per_spk || exit 1;

    $cmd $dir/scoring_kaldi/log/stats2.cer.log \
      cat $dir/scoring_kaldi/cer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring_kaldi/cer_details/ops || exit 1;

    $cmd $dir/scoring_kaldi/log/cer_bootci.cer.log \
      compute-wer-bootci --mode=present \
        ark:$dir/scoring_kaldi/test_filt.chars.txt ark:$dir/scoring_kaldi/penalty_$best_wip/${best_lmwt}.chars.txt \
        '>' $dir/scoring_kaldi/cer_details/cer_bootci || exit 1;

  fi
fi

# If we got here, the scoring was successful.
# As a  small aid to prevent confusion, we remove all wer_{?,??} files;
# these originate from the previous version of the scoring files
# i keep both statement here because it could lead to confusion about
# the capabilities of the script (we don't do cer in the script)
rm $dir/wer_{?,??} 2>/dev/null
rm $dir/cer_{?,??} 2>/dev/null

exit 0;
