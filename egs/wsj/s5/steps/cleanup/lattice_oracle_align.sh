#! /bin/bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

set -e
set -o pipefail

cleanup=true
stage=0
cmd=run.pl
special_symbol="***"    # Special symbol to be aligned with the inserted or
                        # deleted words. Your sentences should not contain this
                        # symbol.
print_silence=true      # True if we want the silences in the ctm.  We do.
frame_shift=0.01

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "This script computes oracle paths for lattices (against a reference "
  echo "transcript) and does various kinds of processing of that, for use by "
  echo "steps/cleanup/cleanup_with_segmentation.sh."
  echo "Its main input is <latdir>/lat.*.gz."
  echo "This script outputs a human-readable word alignment of the oracle path"
  echo "through the lattice in <dir>/oracle_hyp.txt, and a time-aligned ctm version of"
  echo "the same in <dir>/ctm."
  echo "It also creates <dir>/edits.txt (the number of edits per utterance),"
  echo "<dir>/text (which is <data>/text but filtering out any utterances that"
  echo "were not decoded for some reason), and <dir>/length.txt, which is the length"
  echo "of the reference transcript, and <dir>/all_info.txt and <dir>/all_info.sorted.txt"
  echo "which contain all the info in a way that's easier to scan for humans."
  echo "Note: most of this is the same as is done in steps/cleanup/find_bad_utts.sh,"
  echo "except it runs from pre-existing lattices."
  echo ""
  echo "Usage: $0 <data> <lang> <latdir> <dir>"
  echo " e.g.: $0 data/train_si284 data/lang exp/tri4_bad_utts/lats exp/tri4_bad_utts/lattice_oracle"
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>            # config containing options"
  echo "  --cleanup <true|false>            # set this to false to disable cleanup of "
  echo "                                    # temporary files (default: true)"
  echo "  --cmd <command-string>            # how to run jobs (default: run.pl)."
  echo "  --special-symbol <special-symbol> #  Symbol to pad with in insertions and deletions in the"
  echo "                                    # output produced in <dir>/analysis/ (default: '***'"
  echo "  --print-silence <true|false>      # Affects ctm generation; default is true (recommended)"
  echo "  --frame-shift <frame-shift>       # Frame shift in seconds; default: 0.01.  Affects ctm generation."
  exit 1
fi

data=$1
lang=$2
latdir=$3
dir=$4

for f in $lang/oov.int $lang/words.txt $data/text $latdir/lat.1.gz $latdir/num_jobs; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

mkdir -p $dir/log

if [ -e $dir/final.mdl ]; then
  model=$dir/final.mdl
elif [ -e $dir/../final.mdl ]; then
  model=$dir/../final.mdl
else
  echo "$0: expected $dir/final.mdl or $dir/../final.mdl to exist"
  exit 1
fi

nj=$(cat $latdir/num_jobs)
oov=$(cat $lang/oov.int)

utils/split_data.sh --per-utt $data $nj

sdata=$data/split${nj}utt

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_oracle.JOB.log \
    lattice-oracle --write-lattices="ark:|gzip -c > $dir/lat.JOB.gz" \
    "ark:gunzip -c $latdir/lat.JOB.gz |" \
    "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|" \
    ark,t:- \| utils/int2sym.pl -f 2- $lang/words.txt '>' $dir/oracle_hyp.JOB.txt || exit 1;

  echo -n "lattice_oracle_align.sh: overall oracle %WER is: "
  grep 'Overall %WER'  $dir/log/get_oracle.*.log  | \
    perl -e 'while (<>){ if (m: (\d+) / (\d+):) { $x += $1; $y += $2}}  printf("%.2f%%\n", $x*100.0/$y); ' | \
    tee $dir/log/oracle_overall_wer.log

  # the awk commands below are to ensure that partially-written files don't confuse us.
  for x in $(seq $nj); do cat $dir/oracle_hyp.$x.txt; done | awk '{if(NF>=1){print;}}' > $dir/oracle_hyp.txt
  if $cleanup; then
    rm $dir/oracle_hyp.*.txt
  fi
fi

echo $nj > $dir/num_jobs


if [ $stage -le 2 ]; then
  # The following command gets the time-aligned ctm as $dir/ctm.JOB.txt.

  if [ -f $lang/phones/word_boundary.int ]; then
    $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
      set -o pipefail '&&' \
      lattice-align-words $lang/phones/word_boundary.int $model "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt '>' $dir/ctm.JOB || exit 1;
  elif [ -f $lang/phones/align_lexicon.int ]; then
    $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
      set -o pipefail '&&' \
      lattice-align-words-lexicon $lang/phones/align_lexicon.int $model  "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-1best ark:- ark:- \| \
      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt '>' $dir/ctm.JOB || exit 1;
  else
    echo "$0: neither $lang/phones/word_boundary.int nor $lang/phones/align_lexicon.int exists: cannot align."
    exit 1;
  fi
  for j in $(seq $nj); do cat $dir/ctm.$j; done > $dir/ctm
  if $cleanup; then rm $dir/ctm.*; fi
  echo "$0: oracle ctm is in $dir/ctm"
fi


# Stages below are really just to satifsy your curiosity; the output is the same
# as that of find_bad_utts.sh.

if [ $stage -le 3 ]; then
  # in case any utterances failed to align, get filtered copy of $data/text
  utils/filter_scp.pl $dir/oracle_hyp.txt < $data/text  > $dir/text
  cat $dir/text | awk '{print $1, (NF-1);}' > $dir/length.txt

  mkdir -p $dir/analysis

  align-text --special-symbol="$special_symbol"  ark:$dir/text ark:$dir/oracle_hyp.txt  ark,t:- | \
    utils/scoring/wer_per_utt_details.pl --special-symbol "***" > $dir/analysis/per_utt_details.txt

  echo "$0: human-readable alignments are in $dir/analysis/per_utt_details.txt"

  awk '{if ($2 == "#csid") print $1" "($4+$5+$6)}' $dir/analysis/per_utt_details.txt > $dir/edits.txt

  n1=$(wc -l < $dir/edits.txt)
  n2=$(wc -l < $dir/oracle_hyp.txt)
  n3=$(wc -l < $dir/text)
  n4=$(wc -l < $dir/length.txt)
  if [ $n1 -ne $n2 ] || [ $n2 -ne $n3 ] || [ $n3 -ne $n4 ]; then
    echo "$0: mismatch in lengths of files:"
    wc $dir/edits.txt $dir/oracle_hyp.txt $dir/text $dir/length.txt
    exit 1;
  fi

  # note: the format of all_info.txt is:
  # <utterance-id>   <number of errors>  <reference-length>  <decoded-output>   <reference>
  # with the fields separated by tabs, e.g.
  # adg04_sr009_trn 1 	12	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED AT	 SHOW THE GRIDLEY+S TRACK IN BRIGHT ORANGE WITH HORNE+S IN DIM RED

  paste $dir/edits.txt \
      <(awk '{print $2}' $dir/length.txt) \
      <(awk '{$1="";print;}' <$dir/oracle_hyp.txt) \
      <(awk '{$1="";print;}' <$dir/text) > $dir/all_info.txt

  sort -nr -k2 $dir/all_info.txt > $dir/all_info.sorted.txt

  echo "$0: per-utterance details sorted from worst to best utts are in $dir/all_info.sorted.txt"
  echo "$0: format is: utt-id num-errs ref-length decoded-output (tab) reference"
fi

if [ $stage -le 4 ]; then
  ###
  # These stats might help people figure out what is wrong with the data
  # a)human-friendly and machine-parsable alignment in the file per_utt_details.txt
  # b)evaluation of per-speaker performance to possibly find speakers with
  #   distinctive accents/speech disorders and similar
  # c)Global analysis on (Ins/Del/Sub) operation, which might be used to figure
  #   out if there is systematic issue with lexicon, pronunciation or phonetic confusability

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_per_spk_details.pl $data/utt2spk > $dir/analysis/per_spk_details.txt

  echo "$0: per-speaker details are in $dir/analysis/per_spk_details.txt"

  cat $dir/analysis/per_utt_details.txt | \
    utils/scoring/wer_ops_details.pl --special-symbol "$special_symbol" | \
    sort -i -b -k1,1 -k4,4nr -k2,2 -k3,3 > $dir/analysis/ops_details.txt

  echo "$0: per-word statistics [corr,sub,ins,del] are in $dir/analysis/ops_details.txt"
fi

if [ $stage -le 5 ]; then
  echo "$0: obtaining ctm edits"

  $cmd $dir/log/get_ctm_edits.log \
    align-text ark:$dir/oracle_hyp.txt ark:$dir/text ark,t:-  \| \
      steps/cleanup/internal/get_ctm_edits.py --oov=$oov --symbol-table=$lang/words.txt \
       /dev/stdin $dir/ctm $dir/ctm_edits || exit 1

  echo "$0: ctm with edits information appended is in $dir/ctm_edits"
fi
