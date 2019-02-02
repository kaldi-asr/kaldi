#!/bin/bash

min_lmwt=7
max_lmwt=17
word_ins_penalty=0.0,0.5,1.0

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

data_dir=$1
graph_dir=$2
decode_dir=$3

test_para=$decode_dir/scoring_kaldi/test_filt_para.txt
test_para_char=$decode_dir/scoring_kaldi/test_filt_para.chars.txt

cat $data_dir/text | \
  local/combine_line_txt_to_paragraph.py > $test_para

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for LMWT in $(seq $min_lmwt $max_lmwt); do
      mkdir -p $decode_dir/para/penalty_$wip
      cat $decode_dir/scoring_kaldi/penalty_$wip/$LMWT.txt | \
      local/combine_line_txt_to_paragraph.py > $decode_dir/para/penalty_$wip/$LMWT.txt
  done
done

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for LMWT in $(seq $min_lmwt $max_lmwt); do
      compute-wer --text --mode=present \
      ark:$test_para ark:$decode_dir/para/penalty_$wip/$LMWT.txt &> $decode_dir/para/wer_${LMWT}_${wip} || exit 1;
  done
done

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for lmwt in $(seq $min_lmwt $max_lmwt); do
    # adding /dev/null to the command list below forces grep to output the filename
    grep WER $decode_dir/para/wer_${lmwt}_${wip} /dev/null
  done
done | utils/best_wer.sh  >& $decode_dir/para/best_wer || exit 1


cat $decode_dir/scoring_kaldi/test_filt.chars.txt | \
  local/combine_line_txt_to_paragraph.py > $test_para_char

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for LMWT in $(seq $min_lmwt $max_lmwt); do
      mkdir -p $decode_dir/para/penalty_$wip
      cat $decode_dir/scoring_kaldi/penalty_$wip/$LMWT.chars.txt | \
      local/combine_line_txt_to_paragraph.py > $decode_dir/para/penalty_$wip/$LMWT.chars.txt
  done
done

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for LMWT in $(seq $min_lmwt $max_lmwt); do
      compute-wer --text --mode=present \
      ark:$test_para_char ark:$decode_dir/para/penalty_$wip/$LMWT.chars.txt &> $decode_dir/para/cer_${LMWT}_${wip} || exit 1;
  done
done

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for lmwt in $(seq $min_lmwt $max_lmwt); do
    # adding /dev/null to the command list below forces grep to output the filename
    grep WER $decode_dir/para/cer_${lmwt}_${wip} /dev/null
  done
done | utils/best_wer.sh  >& $decode_dir/para/best_cer || exit 1
