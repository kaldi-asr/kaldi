#!/bin/bash

# Copyright 2015  Tom Ko
# Apache 2.0

# Begin configuration section.
frame_shift=10
# End configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "This script concatenates segments with length shorter than the specified length in seconds."
  echo ""
  echo "Usage: steps/cleanup/combine_short_segments.sh [options] <minimum-length> <input-data-dir> <output-data-dir>"
  echo "e.g.:  steps/cleanup/combine_short_segments.sh 2.0 data/train data/train_min2sec"
  echo ""
  echo "Options:"
  echo "    --frame-shift   <frame-shift|10>         # Frame shift in milliseconds.  Only relevant if"
  echo "                                             # <data-in>/segments does not exist (in which case"
  echo "                                             # the lengths are to be worked out from the features)."
  exit 1;
fi

min_seg_len=$1
input_dir=$2
output_dir=$3

for f in spk2utt text utt2spk feats.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

export LC_ALL=C

mkdir -p $output_dir

if [ -f $input_dir/segments ]; then
  awk '{
    len=$4 - $3;
    if (len > 0) {
      printf("%s %.2f\n", $1, len);
    }
  }' $input_dir/segments >$output_dir/feats.length
else
  feat-to-len --print-args=false scp:$input_dir/feats.scp ark,t:- | \
  awk -v fs=$frame_shift '{
    len=$2 * fs / 1000;
    printf("%s %.2f\n", $1, len);
  }' >$output_dir/feats.length
fi

# The following perl script is the core part.
# It looks for segments with length shorter than the specified length
# and concatenates them with other segments to make sure every combined segments are
# with enough length. Here, after the search of the under-length segment, it looks for
# another segment where the combined length is the closest to the specified length.

echo $min_seg_len |  perl -e '
  $min_seg_len = <STDIN>;
  ($u2s_in, $s2u_in, $text_in, $feat_in, $len_in,
    $u2s_out, $text_out, $feat_out) = @ARGV;
  open(UI, "<$u2s_in") || die "Error: fail to open $u2s_in\n";
  open(SI, "<$s2u_in") || die "Error: fail to open $s2u_in\n";
  open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
  open(FI, "<$feat_in") || die "Error: fail to open $feat_in\n";
  open(LI, "<$len_in") || die "Error: fail to open $len_in\n";
  open(UO, ">$u2s_out") || die "Error: fail to open $u2s_out\n";
  open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
  open(FO, ">$feat_out") || die "Error: fail to open $feat_out\n";
  while (<UI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    ($utt_id, $spk) = @col;
    $utt2spk{$utt_id} = $spk;
  }
  while (<SI>) {
    chomp;
    @col = split;
    $spks = join(" ", @col[1..@col-1]);
    $spk2utt{$col[0]} = $spks;
  }
  while (<FI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    ($utt_id, $feat) = @col;
    $uttlist{$utt_id} = $utt_id;
    $utt2feat{$utt_id} = $feat;
  }
  while (<LI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    ($utt_id, $len) = @col;
    $utt2len{$utt_id} = $len;
    $utt2item{$utt_id} = 1;  #utt2item stores no of utterances that combined
  }
  while (<TI>) {
    chomp;
    @col = split;
    $text = join(" ", @col[1..@col-1]);
    $utt2text{$col[0]} = $text;
  }

  # Once the under-length segment is found, we concatenate it with
  # the contiguous segments. We first combine it with segments after it
  # and then with segments before it.
  foreach $seg (sort keys %uttlist) {
    if ($utt2len{$seg} < $min_seg_len && $utt2item{$seg} > 0) {
      $forward_combining = 0;
      $backward_combining = 0;
      @utts = split(" ", $spk2utt{$utt2spk{$seg}});
      foreach $seg2 (sort @utts) {
        if ($seg2 gt $seg && $utt2item{$seg2} > 0) {
          $sum = $utt2len{$seg} + $utt2len{$seg2};
          $utt2len{$seg} = $sum;
          $utt2len{$seg2} = -1;
          $uttlist{$seg} = $uttlist{$seg} . "-" . $uttlist{$seg2};
          $utt2feat{$seg} = $utt2feat{$seg} . " " . $utt2feat{$seg2};
          $utt2text{$seg} = $utt2text{$seg} . " " . $utt2text{$seg2};
          $utt2item{$seg} = $utt2item{$seg} + $utt2item{$seg2};
          $utt2item{$seg2} = 0;
          if ($sum >= $min_seg_len) {
            $forward_combining = 1;
            last;
          }
        }
      }
      if ($forward_combining == 0) {
        foreach $seg2 (reverse sort @utts) {
          if ($seg2 lt $seg && $utt2item{$seg2} > 0) {
            $sum = $utt2len{$seg} + $utt2len{$seg2};
            $utt2len{$seg} = $sum;
            $utt2len{$seg2} = -1;
            $uttlist{$seg} = $uttlist{$seg} . "-" . $uttlist{$seg2};
            $utt2feat{$seg} = $utt2feat{$seg} . " " . $utt2feat{$seg2};
            $utt2text{$seg} = $utt2text{$seg} . " " . $utt2text{$seg2};
            $utt2item{$seg} = $utt2item{$seg} + $utt2item{$seg2};
            $utt2item{$seg2} = 0;
            if ($sum >= $min_seg_len) {
              $backward_combining = 1;
              last;
            }
          }
        }
      }

      if ($forward_combining == 0 && $backward_combining == 0) {
        print "Warning: speaker $utt2spk{$seg} has no enough segments to reach the specified length.\n";
      }
    }
  }

  foreach $seg (sort keys %uttlist) {
    if ($utt2item{$seg} > 1) {
      print UO "$uttlist{$seg}-appended $utt2spk{$seg}\n";
      print FO "$uttlist{$seg}-appended concat-feats --print-args=false $utt2feat{$seg} - |\n";
      print TO "$uttlist{$seg}-appended $utt2text{$seg}\n";
    } elsif ($utt2item{$seg} == 1) {
      print UO "$uttlist{$seg} $utt2spk{$seg}\n";
      print FO "$uttlist{$seg} $utt2feat{$seg}\n" ;
      print TO "$uttlist{$seg} $utt2text{$seg}\n";
    }
  }

' $input_dir/utt2spk $input_dir/spk2utt $input_dir/text $input_dir/feats.scp \
$output_dir/feats.length $output_dir/utt2spk $output_dir/text $output_dir/feats.scp

utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

if [ -f $input_dir/cmvn.scp ]; then
  cp $input_dir/cmvn.scp $output_dir/
fi

rm $output_dir/feats.length

utils/fix_data_dir.sh $output_dir

exit 0
