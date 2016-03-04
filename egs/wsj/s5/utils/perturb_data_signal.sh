#!/bin/bash 

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2014  Tom Ko
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  wav.scp
#  spk2utt
#  utt2spk
#  text
#  spk_filter.scp
# It generates the files which are used for perturbing the data at signal-level.

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: perturb_data_signal.sh <prefix> <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 'fp01' data/train_si284 data/train_si284p"
  exit 1
fi

export LC_ALL=C

prefix=$1
srcdir=$2
destdir=$3
spk_prefix=$prefix"-"
utt_prefix=$prefix"-"

for f in spk2utt text utt2spk wav.scp spk_filter.scp; do
  [ ! -f $srcdir/$f ] && echo "$0: no such file $srcdir/$f" && exit 1;
done

set -e;
set -o pipefail

mkdir -p $destdir

cat $srcdir/utt2spk | awk -v p=$utt_prefix '{printf("%s %s%s\n", $1, p, $1);}' > $destdir/utt_map
cat $srcdir/spk2utt | awk -v p=$spk_prefix '{printf("%s %s%s\n", $1, p, $1);}' > $destdir/spk_map
cat $srcdir/utt2spk | awk -v p=$utt_prefix '{printf("%s%s %s\n", p, $1, $1);}' > $destdir/utt2uniq

cat $srcdir/utt2spk | utils/apply_map.pl -f 1 $destdir/utt_map | \
  utils/apply_map.pl -f 2 $destdir/spk_map >$destdir/utt2spk

utils/utt2spk_to_spk2utt.pl <$destdir/utt2spk >$destdir/spk2utt


# The following perl script is the core part.

echo $spk_prefix | perl -e '
  $prefix = <STDIN>;
  chomp($prefix);
  ($u2s_in, $seg_in, $wav_in, $filt_in, $wav_out) = @ARGV;
  if (open(SEG, "<$seg_in")) {
    $have_segments="true";
  } else {
    $have_segments="false";
  }
  open(UI, "<$u2s_in") || die "Error: fail to open $u2s_in\n";
  open(WI, "<$wav_in") || die "Error: fail to open $wav_in\n";
  open(FI, "<$filt_in") || die "Error: fail to open $filt_in\n";
  open(WO, ">$wav_out") || die "Error: fail to open $wav_out\n";
  while (<UI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    ($utt_id, $spk) = @col;
    $utt2spk{$utt_id} = $spk;
  }
  if ($have_segments eq "true") {
    while (<SEG>) {
      chomp;
      @col = split;
      $reco2utt{$col[1]} = $col[0];
    }
  }
  while (<WI>) {
    chomp;
    @col = split;
    $pipe = join(" ", @col[1..@col-1]);
    $reco2pipe{$col[0]} = $pipe;
    $recolist{$col[0]} = $col[0];
    if ($have_segments eq "false") {
      $reco2utt{$col[0]} = $col[0];
    }
  }
  while (<FI>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    $spk2filt{$col[0]} = $col[1];
  }

  foreach $reco (sort keys %recolist) {
    #$reco2spk{$reco} = $utt2spk{$reco2utt{$reco}};
    #$reco2filt{$reco} = $spk2filt{$utt2spk{$reco2utt{$reco}}};
    $reco2spk{$reco} = $reco;
    $reco2filt{$reco} = $spk2filt{$reco};
    if ($reco2filt{$reco} eq "") {
      $spk = (keys %spk2filt)[rand keys %spk2filt];
      $reco2spk{$reco} = $spk;
      $reco2filt{$reco} = $spk2filt{$spk};
    }
    while (1) {
      # randomly pick a filter from another speaker
      $spk = (keys %spk2filt)[rand keys %spk2filt];
      $reco2perturbspk{$reco} = $spk;
      $reco2perturbfilt{$reco} = $spk2filt{$spk};
      if ($reco2perturbfilt{$reco} ne $reco2filt{$reco}) {
        last;
      }
    }
  }

  foreach $reco (sort keys %recolist) {
    print WO "$prefix$reco $reco2pipe{$reco} apply-filter --inverse=false \"ark:echo $reco2spk{$reco} $reco2filt{$reco} | \" - - | apply-filter --inverse=true \"ark:echo $reco2perturbspk{$reco} $reco2perturbfilt{$reco} | \" - - |\n";
  }

' $srcdir/utt2spk $srcdir/segments $srcdir/wav.scp \
$srcdir/spk_filter.scp $destdir/wav.scp

if [ -f $srcdir/segments ]; then
  # also apply the spk_prefix to the recording-ids.
  cat $srcdir/wav.scp | awk -v p=$spk_prefix '{printf("%s %s%s\n", $1, p, $1);}' > $destdir/reco_map

  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/segments | utils/apply_map.pl -f 2 $destdir/reco_map >$destdir/segments

  if [ -f $srcdir/reco2file_and_channel ]; then
    utils/apply_map.pl -f 1 $destdir/reco_map <$srcdir/reco2file_and_channel >$destdir/reco2file_and_channel
  fi
  
  rm $destdir/reco_map 2>/dev/null
fi

if [ -f $srcdir/text ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/text >$destdir/text
fi
if [ -f $srcdir/spk2gender ]; then
  utils/apply_map.pl -f 1 $destdir/spk_map <$srcdir/spk2gender >$destdir/spk2gender
fi


rm $destdir/spk_map $destdir/utt_map 2>/dev/null
echo "$0: generated signal-perturbed version of data in $srcdir, in $destdir"
utils/validate_data_dir.sh --no-feats $destdir
