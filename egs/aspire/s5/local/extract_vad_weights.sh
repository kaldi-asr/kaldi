#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script converts lattices available from a first pass decode into a per-frame weights file
# The ctms generated from the lattices are filtered. Silence frames are assigned a low weight (e.g.0.00001)
# and voiced frames have a weight of 1.

set -e

stage=1
cmd=run.pl
iter=final
silence_weight=0.00001
#end configuration section.

. ./cmd.sh

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1;
if [ $# -ne 4 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <input-decode-dir> <output-wts-file-gzipped>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data_dir=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
input_decode_dir=$3
output_wts_file_gz=$4

if [ $stage -le 1 ]; then
  echo "$0: generating CTM from input lattices"
  local/multi_condition/get_ctm_conf.sh --cmd "$cmd" \
    --use-segments false \
    --iter $iter \
    $data_dir \
    $lang \
    $input_decode_dir
fi

if [ $stage -le 2 ]; then
  name=`basename $data_dir`
  # we just take the ctm from LMWT 10, it doesn't seem to affect the results a lot
  ctm=$input_decode_dir/score_10/$name.ctm
  echo "$0: generating weights file from ctm $ctm"

  pad_frames=0  # this did not seem to be helpful but leaving it as an option.
  feat-to-len scp:$data_dir/feats.scp ark,t:- >$input_decode_dir/utt.lengths
  if [ ! -f $ctm ]; then  echo "$0: expected ctm to exist: $ctm"; exit 1; fi

  cat $ctm | awk '$6 == 1.0 && $4 < 1.0' | \
  grep -v -w mm | grep -v -w mhm | grep -v -F '[noise]' | \
  grep -v -F '[laughter]' | grep -v -F '<unk>' | \
  perl -e ' $lengths=shift @ARGV;  $pad_frames=shift @ARGV; $silence_weight=shift @ARGV;
   $pad_frames >= 0 || die "bad pad-frames value $pad_frames";
   open(L, "<$lengths") || die "opening lengths file";
   @all_utts = ();
   $utt2ref = { };
   while (<L>) {
     ($utt, $len) = split(" ", $_);
     push @all_utts, $utt;
     $array_ref = [ ];
     for ($n = 0; $n < $len; $n++) { ${$array_ref}[$n] = $silence_weight; }
     $utt2ref{$utt} = $array_ref;
   }
   while (<STDIN>) {
     @A = split(" ", $_);
     @A == 6 || die "bad ctm line $_";
     $utt = $A[0]; $beg = $A[2]; $len = $A[3];
     $beg_int = int($beg * 100) - $pad_frames;
     $len_int = int($len * 100) + 2*$pad_frames;
     $array_ref = $utt2ref{$utt};
     !defined $array_ref  && die "No length info for utterance $utt";
     for ($t = $beg_int; $t < $beg_int + $len_int; $t++) {
       if ($t >= 0 && $t < @$array_ref) {
         ${$array_ref}[$t] = 1;
        }
      }
    }
    foreach $utt (@all_utts) {  $array_ref = $utt2ref{$utt};
      print $utt, " [ ", join(" ", @$array_ref), " ]\n";
      } ' $input_decode_dir/utt.lengths $pad_frames $silence_weight | \
        gzip -c > $output_wts_file_gz
fi
