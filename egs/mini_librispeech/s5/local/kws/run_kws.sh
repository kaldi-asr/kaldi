#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
flen=0.01
stage=0
cmd=run.pl
data=data/dev_clean_2
lang=data/lang
keywords=local/kws/example/keywords.txt
output=data/dev_clean_2/kws/
# End configuration section

. ./utils/parse_options.sh
. ./path.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

mkdir -p $output
if [ $stage -le 1 ] ; then
  ## generate the auxiliary data files
  ## utt.map
  ## wav.map
  ## trials
  ## frame_length
  ## keywords.int

  ## For simplicity, we do not generate the following files
  ## categories

  ## We will generate the following files later
  ## hitlist
  ## keywords.fsts

  [ ! -f $data/utt2dur ] &&
    utils/data/get_utt2dur.sh $data

  duration=$(cat $data/utt2dur | awk '{sum += $2} END{print sum}' )

  echo $duration > $output/trials
  echo $flen > $output/frame_length

  echo "Number of trials: $(cat $output/trials)"
  echo "Frame lengths: $(cat $output/frame_length)"

  echo "Generating map files"
  cat $data/utt2dur | awk 'BEGIN{i=1}; {print $1, i; i+=1;}' > $output/utt.map
  cat $data/wav.scp | awk 'BEGIN{i=1}; {print $1, i; i+=1;}' > $output/wav.map

  cp $lang/words.txt $output/words.txt
  cp $keywords $output/keywords.txt
  cat $output/keywords.txt | \
    local/kws/keywords_to_indices.pl --map-oov 0  $output/words.txt | \
    sort -u > $output/keywords.int
fi

if [ $stage -le 2 ] ; then
  ## this step generates the file hitlist

  ## in many cases, when the reference hits are given, the followin two steps \
  ## are not needed
  ## we create the alignments of the data directory
  ## this is only so that we can obtain the hitlist
  steps/align_fmllr.sh --nj 5 --cmd "$cmd" \
    $data data/lang exp/tri3b exp/tri3b_ali_$(basename $data)

  local/kws/create_hitlist.sh $data $lang data/local/lang_tmp \
    exp/tri3b_ali_$(basename $data) $output
fi

if [ $stage -le 3 ] ; then
  ## this steps generates the file keywords.fsts

  ## compile the keywords (it's done via tmp work dirs, so that
  ## you can use the keywords filtering and then just run fsts-union
  local/kws/compile_keywords.sh $output $lang  $output/tmp.2
  cp $output/tmp.2/keywords.fsts $output/keywords.fsts
  # for example
  #    fsts-union scp:<(sort data/$dir/kwset_${set}/tmp*/keywords.scp) \
  #      ark,t:"|gzip -c >data/$dir/kwset_${set}/keywords.fsts.gz"
  ##
fi

system=exp/chain/tdnn1h_sp_online/decode_tglarge_dev_clean_2/
if [ $stage -le 4 ]; then
  ## this is not exactly necessary for a single system and single keyword set
  ## but if you have multiple keyword sets, then it avoids having to recompute
  ## the indices unnecesarily every time (see --indices-dir and --skip-indexing
  ## parameters to the search script bellow).
  for lmwt in `seq 8 14` ; do
    steps/make_index.sh --cmd "$cmd" --lmwt $lmwt --acwt 1.0 \
      --frame-subsampling-factor 3\
      $output $lang $system $system/kws_indices_$lmwt
  done
fi

if [ $stage -le 5 ]; then
  ## find the hits, normalize and score
  local/kws/search.sh --cmd "$cmd" --min-lmwt 8 --max-lmwt 14  \
    --indices-dir $system/kws_indices --skip-indexing true\
    $lang $data $system
fi

echo "Done"


