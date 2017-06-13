#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.
# 2014, University of Edinburgh, (Author: Pawel Swietojanski)
# 2015, Brno University of Technology (Author: Karel Vesely)

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
min_lmwt=7
max_lmwt=15
asclite=true
iter=final
overlap_spk=4
# end configuration section.
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_asclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$dir/../$iter.mdl # assume model one level up from decoding dir.

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

if [ -f $dir/../frame_shift ]; then
  frame_shift_opt="--frame-shift=$(cat $dir/../frame_shift)"
  echo "$0: $dir/../frame_shift exists, using $frame_shift_opt"
elif [ -f $dir/../frame_subsampling_factor ]; then
  factor=$(cat $dir/../frame_subsampling_factor) || exit 1
  frame_shift_opt="--frame-shift=0.0$factor"
  echo "$0: $dir/../frame_subsampling_factor exists, using $frame_shift_opt"
fi

name=`basename $data`; # e.g. eval2000
nj=$(cat $dir/num_jobs)

mkdir -p $dir/ascoring/log

if [ $stage -le 0 ]; then
  for LMWT in $(seq $min_lmwt $max_lmwt); do
    rm -f $dir/.error
    (
    $cmd JOB=1:$nj $dir/ascoring/log/get_ctm.${LMWT}.JOB.log \
      mkdir -p $dir/ascore_${LMWT}/ '&&' \
      lattice-scale --inv-acoustic-scale=${LMWT} "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-limit-depth ark:- ark:- \| \
      lattice-push --push-strings=false ark:- ark:- \| \
      lattice-align-words-lexicon --max-expand=10.0 \
       $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      lattice-to-ctm-conf $frame_shift_opt --decode-mbr=$decode_mbr ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/ascore_${LMWT}/${name}.JOB.ctm || touch $dir/.error;
    # Merge and clean,
    for ((n=1; n<=nj; n++)); do cat $dir/ascore_${LMWT}/${name}.${n}.ctm; done > $dir/ascore_${LMWT}/${name}.ctm
    rm -f $dir/ascore_${LMWT}/${name}.*.ctm
    )&
  done
  wait;
  [ -f $dir/.error ] && echo "$0: error during ctm generation. check $dir/ascoring/log/get_ctm.*.log" && exit 1;
fi

if [ $stage -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
# - we remove hesitations here, otherwise the CTM would have a bug!
#   (confidences in place of the removed hesitations),
  for x in $dir/ascore_*/${name}.ctm; do
    cp $x $x.tmpf;
    cat $x.tmpf | grep -i -v -E '\[noise|laughter|vocalized-noise\]' | \
      grep -i -v -E ' (ACH|AH|EEE|EH|ER|EW|HA|HEE|HM|HMM|HUH|MM|OOF|UH|UM) ' | \
      grep -i -v -E '<unk>' > $x;
#      grep -i -v -E '<UNK>|%HESITATION' > $x;
  done
fi

if [ $stage -le 2 ]; then
  if [ "$asclite" == "true" ]; then
    oname=$name
    [ ! -z $overlap_spk ] && oname=${name}_o$overlap_spk
    echo "asclite is starting"
    # Run scoring, meaning of hubscr.pl options:
    # -G .. produce alignment graphs,
    # -v .. verbose,
    # -m .. max-memory in GBs,
    # -o .. max N of overlapping speakers,
    # -a .. use asclite,
    # -C .. compression for asclite,
    # -B .. blocksize for asclite (kBs?),
    # -p .. path for other components,
    # -V .. skip validation of input transcripts,
    # -h rt-stt .. removes non-lexical items from CTM,
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
      cp $data/stm $dir/ascore_LMWT/ '&&' \
      cp $dir/ascore_LMWT/${name}.ctm $dir/ascore_LMWT/${oname}.ctm '&&' \
      $hubscr -G -v -m 1:2 -o$overlap_spk -a -C -B 8192 -p $hubdir -V -l english \
        -h rt-stt -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${oname}.ctm || exit 1
    # Compress some scoring outputs : alignment info and graphs,
    echo -n "compressing asclite outputs "
    for LMWT in $(seq $min_lmwt $max_lmwt); do
      ascore=$dir/ascore_${LMWT}
      gzip -f $ascore/${oname}.ctm.filt.aligninfo.csv
      cp $ascore/${oname}.ctm.filt.alignments/index.html $ascore/${oname}.ctm.filt.overlap.html
      tar -C $ascore -czf $ascore/${oname}.ctm.filt.alignments.tar.gz ${oname}.ctm.filt.alignments
      rm -r $ascore/${oname}.ctm.filt.alignments
      echo -n "LMWT:$LMWT "
    done
    echo done
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
      cp $data/stm $dir/ascore_LMWT/ '&&' \
      $hubscr -p $hubdir -v -V -l english -h hub5 -g $data/glm -r $dir/ascore_LMWT/stm $dir/ascore_LMWT/${name}.ctm || exit 1
  fi
fi

exit 0
