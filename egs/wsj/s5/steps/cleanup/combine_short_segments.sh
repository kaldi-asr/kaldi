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
  echo "Usage: steps/cleanup/combine_short_segments.sh [options] <minimum-length> <data-in> <data-out>"
  echo "e.g.:  steps/cleanup/combine_short_segments.sh 2.0 data/train data/train_min2sec"
  echo ""
  echo "Options:"
  echo "    --frame-shift   <frame-shift|10>         # Frame shift in milliseconds.  Only relevant if"
  echo "                                             # <data-in>/segments does not exist (in which case"
  echo "                                             # the lengths are to be worked out from the features)."
  exit 1;
fi

#This script works in the unit of frames
frame_per_sec=$(( 1000/$frame_shift ))
min_seg_len=$(echo $1*$frame_per_sec | bc)
input_dir=$2
output_dir=$3

for f in spk2utt text utt2spk feats.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

export LC_ALL=C

# This is the main function, to look for segments with length shorter than the specified length
# and concatenates them with other segments to make sure every combined segments are 
# with enough length. Here, after the search of the under-length segment, it looks for
# another segment where the combined length is the closest to the specified length.

function check_and_combine_utt {

echo $min_seg_len |  perl -e '
  $min_seg_len = <STDIN>;
  ($u2s_in, $s2u_in, $feat_in, $len_in, $text_in, 
    $u2s_out, $feat_out, $text_out) = @ARGV;
  open(UI, "<$u2s_in") || die "Error: fail to open $u2s_in\n";
  open(SI, "<$s2u_in") || die "Error: fail to open $s2u_in\n";
  open(FI, "<$feat_in") || die "Error: fail to open $feat_in\n";
  open(LI, "<$len_in") || die "Error: fail to open $len_in\n";
  open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
  open(UO, ">$u2s_out") || die "Error: fail to open $u2s_out\n";
  open(FO, ">$feat_out") || die "Error: fail to open $feat_out\n";
  open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
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
    $utt2item{$utt_id} = 1;
  }
  while (<TI>) {
    chomp;
    @col = split;
    $text = join(" ", @col[1..@col-1]);
    $utt2text{$col[0]} = $text;
  }

  foreach $seg (sort keys %uttlist) {
    if ($utt2len{$seg} < $min_seg_len) {
      $localmin = inf;
      $target = $seg;
      @utts = split(" ",$spk2utt{$utt2spk{$seg}});
      foreach $seg2 (@utts) {
        $sum = $utt2len{$seg} + $utt2len{$seg2};
        if (($sum >= $min_seg_len) && ($sum < $localmin) && ($seg ne $seg2)) {
          $localmin = $sum;
          $target = $seg2;
        }
      }
      if ($target eq $seg) { die "Error: fail to have combined segment with enough length\n"; }
      $utt2len{$seg} = -1;
      $uttlist{$target} = $uttlist{$target} . "-" . $uttlist{$seg};
      $utt2len{$target} = $localmin;
      $utt2feat{$target} = $utt2feat{$target} . " " . $utt2feat{$seg};
      $utt2item{$target} = $utt2item{$target} + $utt2item{$seg};
      $utt2item{$seg} = 0;
      $utt2text{$target} = $utt2text{$target} . " " . $utt2text{$seg};
    }
  }

  foreach $seg (sort keys %uttlist) {
    if ($utt2item{$seg} > 1) {
      print UO "$uttlist{$seg}-appended $utt2spk{$seg}\n";
      print FO "$uttlist{$seg}-appended concat-feats $utt2feat{$seg} - |\n";
      print TO "$uttlist{$seg}-appended $utt2text{$seg}\n";
    } elsif ($utt2item{$seg} == 1) {
      print UO "$uttlist{$seg} $utt2spk{$seg}\n";
      print FO "$uttlist{$seg} $utt2feat{$seg}\n" ;
      print TO "$uttlist{$seg} $utt2text{$seg}\n";
    }
  }

' $input_dir/utt2spk \
$input_dir/spk2utt \
$input_dir/feats.scp \
$output_dir/feats.length \
$input_dir/text \
$output_dir/utt2spk \
$output_dir/feats.scp \
$output_dir/text 
}

mkdir -p $output_dir

if [ -f $input_dir/segments ]; then
  awk -v l=$frame_per_sec '{
    len=($4 - $3) * l;
    if (len > 0) {
      printf("%s %d \n", $1, len);
    }
  }' $input_dir/segments >$output_dir/feats.length
else
  feat-to-len --print-args=false scp:$input_dir/feats.scp ark,t:$output_dir/feats.length
fi

check_and_combine_utt

utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

if [ -f $input_dir/cmvn.scp ]; then
  cp $input_dir/cmvn.scp $output_dir/
fi

rm $output_dir/feats.length

utils/fix_data_dir.sh $output_dir

exit 0
