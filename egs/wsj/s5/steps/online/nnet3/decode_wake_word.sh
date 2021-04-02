#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
#           2016  Api.ai (Author: Ilya Platonov)
#      2019-2020  Yiming Wang
# Apache 2.0

# This script is modified from steps/online/nnet3/decode.sh for wake word detection decoding

# Begin configuration section.
stage=0
nj=4
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
cmd=run.pl
frames_per_chunk=20
extra_left_context_initial=0
min_active=200
max_active=7000
beam=15.0
per_utt=false
online=true  # only relevant to non-threaded decoder.
do_speex_compressing=false
scoring_opts=
skip_scoring=false
iter=final
online_config=
wake_word="嗨小问"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the models are, as prepared by steps/online/nnet3/prepare_online_decoding.sh"
   echo "e.g.: $0 exp/chain/tdnn/graph data/test exp/chain/tdnn_online/decode/"
   echo ""
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --online-config <config-file>                    # online decoder options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --per-utt <true|false>                           # If true, decode per utterance without"
   echo "                                                   # carrying forward adaptation info from previous"
   echo "                                                   # utterances of each speaker.  Default: false"
   echo "  --online <true|false>                            # Set this to false if you don't really care about"
   echo "                                                   # simulating online decoding and just want the best"
   echo "                                                   # results.  This will use all the data within each"
   echo "                                                   # utterance (plus any previous utterance, if not in"
   echo "                                                   # per-utterance mode) to estimate the iVectors."
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --iter <iter>                                    # Iteration of model to decode; default is final."
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

if [ "$online_config" == "" ]; then
  online_config=$srcdir/conf/online.conf;
fi

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

for f in $online_config $srcdir/${iter}.mdl \
    $graphdir/HCLG.fst $graphdir/words.txt $data/wav.scp; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if ! $per_utt; then
  spk2utt_rspecifier="ark:$sdata/JOB/spk2utt"
else
  mkdir -p $dir/per_utt
  for j in $(seq $nj); do
    awk '{print $1, $1}' <$sdata/$j/utt2spk >$dir/per_utt/utt2spk.$j || exit 1;
  done
  spk2utt_rspecifier="ark:$dir/per_utt/utt2spk.JOB"
fi

if [ -f $data/segments ]; then
  wav_rspecifier="ark,s,cs:extract-segments scp,p:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- |"
else
  wav_rspecifier="ark,s,cs:wav-copy scp,p:$sdata/JOB/wav.scp ark:- |"
fi
if $do_speex_compressing; then
  wav_rspecifier="$wav_rspecifier compress-uncompress-speex ark:- ark:- |"
fi

wake_word_id=$(cat $graphdir/words.txt | grep $wake_word | awk '{print $2}')

if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
fi

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    online2-wav-nnet3-wake-word-decoder-faster \
    --frames-per-chunk=$frames_per_chunk \
    --extra-left-context-initial=$extra_left_context_initial \
    --online=$online \
       $frame_subsampling_opt \
     --config=$online_config \
     --min-active=$min_active --max-active=$max_active --beam=$beam \
     --acoustic-scale=$acwt --wake-word-id=$wake_word_id \
     $srcdir/${iter}.mdl $graphdir/HCLG.fst $spk2utt_rspecifier "$wav_rspecifier" \
     $graphdir/words.txt ark,t:$dir/trans.JOB.txt \
     ark,t:$dir/ali.JOB.txt || exit 1;
fi

if [ $stage -le 1 ]; then
  for n in $(seq $nj); do
    cat $dir/trans.$n.txt
  done > $dir/trans.txt
  rm -f $dir/trans.*.txt
  for n in $(seq $nj); do
    cat $dir/ali.$n.txt
  done > $dir/ali.txt
  rm -f $dir/ali.*.txt
fi

if [ $stage -le 2 ] && ! $skip_scoring ; then
  [ ! -x local/score_online.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score_online.sh $scoring_opts --wake-word $wake_word $data $graphdir $dir
fi

exit 0;
