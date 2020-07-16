#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments using a model with delta or
# LDA+MLLT features.  This version, rather than just using the
# text to align, computes mini-language models (unigram) from the text
# and a few common words in the LM.

# Begin configuration section.
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
acoustic_scale=0.1
beam=15.0
lattice_beam=8.0
max_active=750
transform_dir=  # directory to find fMLLR transforms in.
top_n_words=100 # Number of common words that we compile into each graph (most frequent
                # in $lang/text.
stage=-1
cleanup=true
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "$0: Warning: this script is deprecated and will be removed."
  echo "  ... please use steps/cleanup/clean_and_segment_data.sh,"
  echo " which produces the same output formats as this script"
  echo " (e.g. all_info.sorted.txt)"
  echo "Usage: $0 <data-dir> <lang-dir> <src-dir> <dir>"
  echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_debug"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --use-graphs true                                # use graphs in src-dir"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

for f in $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl \
    $lang/L_disambig.fst $lang/phones/disambig.int; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;


if [ $stage -le 0 ]; then
  utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt <$data/text | \
    awk '{for(x=2;x<=NF;x++) print $x;}' | sort | uniq -c | \
    sort -rn > $dir/word_counts.int || exit 1;
  num_words=$(awk '{x+=$1} END{print x}' < $dir/word_counts.int) || exit 1;
  # print top-n words with their unigram probabilities.

  head -n $top_n_words $dir/word_counts.int | awk -v tot=$num_words '{print $1/tot, $2;}' >$dir/top_words.int
  utils/int2sym.pl -f 2 $lang/words.txt <$dir/top_words.int >$dir/top_words.txt
fi

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $srcdir/full.mat $dir
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -z "$transform_dir" ] && [ -f $srcdir/trans.1 ]; then
  transform_dir=$srcdir
fi
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
      copy-feats ark:- ark,scp:$dir/trans.ark,$dir/trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
  fi
elif [ -f $srcdir/final.alimdl ]; then
  echo "$0: **WARNING**: you seem to be using an fMLLR system as input,"
  echo "  but you are not providing the --transform-dir option during alignment."
fi


if [ $stage -le 1 ]; then
  echo "$0: decoding $data using utterance-specific decoding graphs using model from $srcdir, output in $dir"

  rm $dir/edits.*.txt $dir/aligned_ref.*.txt 2>/dev/null

  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text \| \
    steps/cleanup/make_utterance_fsts.pl $dir/top_words.int \| \
    compile-train-graphs-fsts $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
     $dir/tree $dir/final.mdl $lang/L_disambig.fst ark:- ark:- \| \
    gmm-latgen-faster --acoustic-scale=$acoustic_scale --beam=$beam \
      --max-active=$max_active --lattice-beam=$lattice_beam \
      --word-symbol-table=$lang/words.txt \
     $dir/final.mdl ark:- "$feats" ark:- \| \
    lattice-oracle ark:- "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|" \
      ark,t:- ark,t:$dir/edits.JOB.txt \| \
    utils/int2sym.pl -f 2- $lang/words.txt '>' $dir/aligned_ref.JOB.txt || exit 1;
fi


if [ $stage -le 2 ]; then
  if [ -f $dir/edits.1.txt ]; then
    # the awk commands below are to ensure that partially-written files don't confuse us.
    for x in $(seq $nj); do cat $dir/edits.$x.txt; done | awk '{if(NF==2){print;}}' > $dir/edits.txt
    for x in $(seq $nj); do cat $dir/aligned_ref.$x.txt; done | awk '{if(NF>=1){print;}}' > $dir/aligned_ref.txt
  else
    echo "$0: warning: no file $dir/edits.1.txt, using previously concatenated file if present."
  fi

  # in case any utterances failed to align, get filtered copy of $data/text
  utils/filter_scp.pl $dir/edits.txt < $data/text  > $dir/text
  cat $dir/text | awk '{print $1, (NF-1);}' > $dir/length.txt

  n1=$(wc -l < $dir/edits.txt)
  n2=$(wc -l < $dir/aligned_ref.txt)
  n3=$(wc -l < $dir/text)
  n4=$(wc -l < $dir/length.txt)
  if [ $n1 -ne $n2 ] || [ $n2 -ne $n3 ] || [ $n3 -ne $n4 ]; then
    echo "$0: mismatch in lengths of files:"
    wc $dir/edits.txt $dir/aligned_ref.txt $dir/text $dir/length.txt
    exit 1;
  fi

  # note: the format of all_info.txt is:
  # <utterance-id>   <number of errors>  <reference-length>  <decoded-output>   <reference>
  # with the fields separated by tabs, e.g.
  # adg04_sr009_trn 1 	12	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED AT	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED

  paste $dir/edits.txt \
      <(awk '{print $2}' $dir/length.txt) \
      <(awk '{$1="";print;}' <$dir/aligned_ref.txt) \
      <(awk '{$1="";print;}' <$dir/text) > $dir/all_info.txt

  sort -nr -k2 $dir/all_info.txt > $dir/all_info.sorted.txt

  if $cleanup; then
    rm $dir/edits.*.txt $dir/aligned_ref.*.txt
  fi

fi

if [ $stage -le 3 ]; then
  ###
  # These stats might help people figure out what is wrong with the data
  # a)human-friendly and machine-parsable alignment in the file per_utt_details.txt
  # b)evaluation of per-speaker performance to possibly find speakers with
  #   distinctive accents/speech disorders and similar
  # c)Global analysis on (Ins/Del/Sub) operation, which might be used to figure
  #   out if there is systematic issue with lexicon, pronunciation or phonetic confusability

  mkdir -p $dir/analysis
  align-text --special-symbol="***"  ark:$dir/text ark:$dir/aligned_ref.txt  ark,t:- | \
    utils/scoring/wer_per_utt_details.pl --special-symbol "***" > $dir/analysis/per_utt_details.txt

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_per_spk_details.pl $data/utt2spk > $dir/analysis/per_spk_details.txt

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_ops_details.pl --special-symbol "***" | \
    sort -i -b -k1,1 -k4,4nr -k2,2 -k3,3 > $dir/analysis/ops_details.txt

fi

