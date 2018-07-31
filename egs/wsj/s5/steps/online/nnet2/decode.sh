#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.
stage=0
nj=4
cmd=run.pl
max_active=7000
threaded=false
modify_ivector_config=false #  only relevant to threaded decoder.
beam=15.0
lattice_beam=6.0
acwt=0.1   # note: only really affects adaptation and pruning (scoring is on
           # lattices).
per_utt=false
online=true  # only relevant to non-threaded decoder.
do_endpointing=false
do_speex_compressing=false
scoring_opts=
skip_scoring=false
silence_weight=1.0  # set this to a value less than 1 (e.g. 0) to enable silence weighting.
max_state_duration=40 # This only has an effect if you are doing silence
  # weighting.  This default is probably reasonable.  transition-ids repeated
  # more than this many times in an alignment are treated as silence.
iter=final
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the models are, as prepared by steps/online/nnet2/prepare_online_decoding.sh"
   echo "e.g.: $0 exp/tri3b/graph data/test exp/tri3b_online/decode/"
   echo ""
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
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
if $per_utt; then
  utt_suffix=utt
  utt_opt="--per-utt"
else
  utt_suffix=
  utt_opt=
fi
sdata=$data/split${nj}${utt_suffix};

mkdir -p $dir/log
split_data.sh $utt_opt $data $nj || exit 1;
echo $nj > $dir/num_jobs

for f in $srcdir/conf/online_nnet2_decoding.conf $srcdir/${iter}.mdl \
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
if $do_endpointing; then
  wav_rspecifier="$wav_rspecifier extend-wav-with-silence ark:- ark:- |"
fi

if [ "$silence_weight" != "1.0" ]; then
  silphones=$(cat $graphdir/phones/silence.csl) || exit 1
  silence_weighting_opts="--ivector-silence-weighting.max-state-duration=$max_state_duration --ivector-silence-weighting.silence_phones=$silphones --ivector-silence-weighting.silence-weight=$silence_weight"
else
  silence_weighting_opts=
fi


if $threaded; then
  decoder=online2-wav-nnet2-latgen-threaded
    # note: the decoder actually uses 4 threads, but the average usage will normally
    # be more like 2.
  parallel_opts="--num-threads 2"
  opts="--modify-ivector-config=$modify_ivector_config --verbose=1"
else
  decoder=online2-wav-nnet2-latgen-faster
  parallel_opts=
  opts="--online=$online"
fi

if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    $decoder $opts $silence_weighting_opts --do-endpointing=$do_endpointing \
     --config=$srcdir/conf/online_nnet2_decoding.conf \
     --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
     --acoustic-scale=$acwt --word-symbol-table=$graphdir/words.txt \
     $srcdir/${iter}.mdl $graphdir/HCLG.fst $spk2utt_rspecifier "$wav_rspecifier" \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir
fi

exit 0;
