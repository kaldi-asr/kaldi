#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh

# Global options,
graph=exp/ihm/tri4a/graph_ami_fsh.o3g.kn.pr1-7
arpa_gz=data/local/lm/ami_fsh.o3g.kn.pr1-7.gz
lmwt=12

# Dev-set options,
dev_data=data/ihm/dev
dev_latdir=exp/ihm/tri4a_mmi_b0.1/decode_dev_4.mdl_ami_fsh.o3g.kn.pr1-7

# Eval-set options,
eval_data=data/ihm/eval
eval_latdir=exp/ihm/tri4a_mmi_b0.1/decode_eval_4.mdl_ami_fsh.o3g.kn.pr1-7

. utils/parse_options.sh
set -euxo pipefail

# Derived options,
dev_caldir=$dev_latdir/confidence_$lmwt
eval_caldir=$eval_latdir/confidence_$lmwt

###### Data preparation,

# Prepare filtering for excluding data from train-set (1 .. keep word, 0 .. exclude word),
# - only excludes from training-targets, the confidences are recalibrated for all the words,
word_filter=$(mktemp)
awk '{ keep_the_word = $1 !~ /^(\[.*\]|<.*>|%.*|!.*|-.*|.*-)$/; print $0, keep_the_word }' \
  $graph/words.txt >$word_filter

# Calcualte the word-length,
word_length=$(mktemp)
awk '{if(r==0) { len_hash[$1] = NF-2; } 
      if(r==1) { if(len_hash[$1]) { len = len_hash[$1]; } else { len = -1 }  
      print $0, len; }}' \
  r=0 $graph/phones/align_lexicon.txt \
  r=1 $graph/words.txt \
  >$word_length

# Extract unigrams,
unigrams=$(mktemp); steps/conf/parse_arpa_unigrams.py $graph/words.txt $arpa_gz $unigrams

###### Paste the 'word-specific' features (first 4 columns have fixed position, more feature-columns can be added),
# Format: "word word_id filter length other_features"
word_feats=$(mktemp)
paste $word_filter <(awk '{ print $3 }' $word_length) <(awk '{ print $3 }' $unigrams) > $word_feats


###### Train the calibration,
steps/conf/train_calibration.sh --cmd "$decode_cmd" --lmwt $lmwt \
  $dev_data $graph $word_feats $dev_latdir $dev_caldir

###### Apply the calibration to eval set,
steps/conf/apply_calibration.sh --cmd "$decode_cmd" \
  $eval_data $graph $eval_latdir $dev_caldir $eval_caldir
# The final confidences are here '$eval_caldir/ctm_calibrated',

###### Sclite scoring,
# We will produce NCE which shows the ``quality'' of the confidences.
# Please compare with the default scoring script for your database.

# Scoring tools,
hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
hubdir=`dirname $hubscr`

# Inputs,
ctm=$eval_caldir/ctm_calibrated
stm=$eval_data/stm
glm=$eval_data/glm

# Normalizng CTM, just like in 'local/score_sclite.sh',
cat $ctm | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
  grep -i -v -E ' (ACH|AH|EEE|EH|ER|EW|HA|HEE|HM|HMM|HUH|MM|OOF|UH|UM) ' | \
  grep -i -v -E '<unk>' >${ctm}.filt

# Mapping the time info to global,
utils/convert_ctm.pl $eval_data/segments $eval_data/reco2file_and_channel <${ctm}.filt >${ctm}.filt.conv

# Scoring,
$hubscr -p $hubdir -V -l english -h hub5 -g $glm -r $stm ${ctm}.filt.conv
