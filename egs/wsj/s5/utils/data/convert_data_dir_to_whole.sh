#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

# This scripts converts a data directory into a "whole" data directory
# by removing the segments and using the recordings themselves as 
# utterances

set -o pipefail

. path.sh

cmd=run.pl

. parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: convert_data_dir_to_whole.sh <in-data> <out-data>"
  echo " e.g.: convert_data_dir_to_whole.sh data/dev data/dev_whole"
  exit 1
fi

data=$1
dir=$2

if [ ! -f $data/segments ]; then
  echo "$0: Data directory already does not contain segments. So just copying it."
  utils/copy_data_dir.sh $data $dir
  exit 0
fi

mkdir -p $dir
cp $data/wav.scp $dir
cp $data/reco2file_and_channel $dir
rm -f $dir/{utt2spk,text} || true

[ -f $data/stm ] && cp $data/stm $dir
[ -f $data/glm ] && cp $data/glm $dir

text_files=
[ -f $data/text ] && text_files="$data/text $dir/text"

# Combine utt2spk and text from the segments into utt2spk and text for the whole
# recording.
cat $data/segments | perl -e '
if (scalar @ARGV == 3) {
  ($utt2spk_in, $text_in, $text_out) = @ARGV;
} elsif (scalar @ARGV == 1) {
  $utt2spk_in = $ARGV[0];
} else {
  die "Unexpected number of arguments";
}

if (defined $text_in) {
  open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
  open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
}
open(UI, "<$utt2spk_in") || die "Error: fail to open $utt2spk_in\n";

my %file2utt = ();
while (<STDIN>) {
  chomp;
  my @col = split;
  @col >= 4 or die "bad line $_\n";

  if (! defined $file2utt{$col[1]}) {
    $file2utt{$col[1]} = [];
  }
  push @{$file2utt{$col[1]}}, $col[0]; 
}

my %text = ();
my %utt2spk = ();

while (<UI>) {
  chomp; 
  my @col = split;
  $utt2spk{$col[0]} = $col[1];
}

if (defined $text_in) {
  while (<TI>) {
    chomp;
    my @col = split;
    @col >= 1 or die "bad line $_\n";

    my $utt = shift @col;
    $text{$utt} = join(" ", @col);
  }
}

foreach $file (keys %file2utt) {
  my @utts = @{$file2utt{$file}};
  print "$file $file\n";

  if (defined $text_in) {
    $text_line = "";
    print TO "$file $text_line\n";
  }
}
' $data/utt2spk $text_files > $dir/utt2spk

utils/spk2utt_to_utt2spk.pl $dir/utt2spk > $dir/spk2utt

utils/fix_data_dir.sh $dir
