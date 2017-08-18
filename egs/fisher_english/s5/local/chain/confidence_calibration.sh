#!/bin/bash
. cmd.sh
. path.sh

chaindir=exp/chain_semi350k_conf/tdnn_xxsup1a_sp
arpa_gz=data/local/lm_ex250k/3gram-mincount/lm_unpruned.gz
graph_affix=_ex250k
decode_affix=
train_set=train_sup_5k_calib_train
dev_set=dev_sup_5k_calib_dev

. utils/parse_options.sh

set -euxo pipefail

train_data=data/${train_set}_hires
dev_data=data/${dev_set}_hires

decode_affix=${decode_affix}${graph_affix}
graphdir=$chaindir/graph${graph_affix}
train_caldir=$chaindir/decode_${train_set}${decode_affix}/confidence
dev_caldir=$chaindir/decode_${dev_set}${decode_affix}/confidence

###### Data preparation,

# Prepare filtering for excluding data from train-set (1 .. keep word, 0 .. exclude word),
# - only excludes from training-targets, the confidences are recalibrated for all the words,
word_filter=$(mktemp)
awk '{ keep_the_word = $1 !~ /^(\[.*\]|<.*>|%.*|!.*|-.*|.*-)$/; print $0, keep_the_word }' \
  $graphdir/words.txt >$word_filter

# Calcualte the word-length,
word_length=$(mktemp)
awk '{if(r==0) { len_hash[$1] = NF-2; } 
      if(r==1) { if(len_hash[$1]) { len = len_hash[$1]; } else { len = -1 }  
      print $0, len; }}' \
  r=0 $graphdir/phones/align_lexicon.txt \
  r=1 $graphdir/words.txt \
  >$word_length

# Extract unigrams,
unigrams=$(mktemp); steps/conf/parse_arpa_unigrams.py $graphdir/words.txt $arpa_gz $unigrams

###### Paste the 'word-specific' features (first 4 columns have fixed position, more feature-columns can be added),
# Format: "word word_id filter length other_features"
word_feats=$(mktemp)
paste $word_filter <(awk '{ print $3 }' $word_length) <(awk '{ print $3 }' $unigrams) > $word_feats


###### Train the calibration,
steps/conf/train_calibration.sh --cmd "$decode_cmd" --lmwt 10 \
  $train_data $graphdir $word_feats \
  $chaindir/decode_${train_set}${decode_affix} $train_caldir

###### Apply the calibration to eval set,
steps/conf/apply_calibration.sh --cmd "$decode_cmd" \
  $dev_data $graphdir $chaindir/decode_${dev_set}${decode_affix} \
  $train_caldir $dev_caldir
# The final confidences are here '$eval_caldir/ctm_calibrated',

exit 0

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
  grep -i -v -E '<UNK>' | \
  grep -i -v -E ' (UH|UM|EH|MM|HM|AH|HUH|HA|ER|OOF|HEE|ACH|EEE|EW) ' | \
  awk '$5 !~ /^.*-$/' | \
  local/map_acronyms_ctm.py -M data/local/dict_nosp/acronyms.map -i - -o ${ctm}.filt

# Mapping the time info to global,
utils/convert_ctm.pl $eval_data/segments $eval_data/reco2file_and_channel <${ctm}.filt >${ctm}.filt.conv

# Scoring,
$hubscr -p $hubdir -V -l english -h hub5 -g $glm -r $stm ${ctm}.filt.conv

