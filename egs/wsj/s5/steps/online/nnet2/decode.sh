#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.  
stage=0
nj=4
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.1   # note: only really affects adaptation and pruning (scoring is on
           # lattices).
per_utt=false
do_endpointing=false
do_speex_compressing=false
scoring_opts=
skip_scoring=false
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
   echo "                                                   # utterances of each speaker."
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

for f in $srcdir/conf/online_nnet2_decoding.conf $srcdir/final.mdl \
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

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    online2-wav-nnet2-latgen-faster --do-endpointing=$do_endpointing \
     --config=$srcdir/conf/online_nnet2_decoding.conf \
     --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
     --acoustic-scale=$acwt --word-symbol-table=$graphdir/words.txt \
     $srcdir/final.mdl $graphdir/HCLG.fst $spk2utt_rspecifier "$wav_rspecifier" \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir
fi

exit 0;
