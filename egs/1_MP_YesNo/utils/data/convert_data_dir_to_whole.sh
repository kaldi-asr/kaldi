#! /bin/bash

# This scripts converts a data directory into a "whole" data directory
# by removing the segments and using the recordings themselves as 
# utterances

set -o pipefail

. path.sh

cmd=run.pl
stage=-1

. parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: convert_data_dir_to_whole.sh <in-data> <out-data>"
  echo " e.g.: convert_data_dir_to_whole.sh data/dev data/dev_whole"
  exit 1
fi

data=$1
dir=$2

if [ ! -f $data/segments ]; then
  # Data directory already does not contain segments. So just copy it.
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
cat $data/segments | sort -k2,2 -k3,4n | perl -e '
if (scalar @ARGV == 4) {
  ($utt2spk_in, $utt2spk_out, $text_in, $text_out) = @ARGV;
} elsif (scalar @ARGV == 2) {
  ($utt2spk_in, $utt2spk_out) = @ARGV;
} else {
  die "Unexpected number of arguments";
}

if (defined $text_in) {
  open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
  open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
}
open(UI, "<$utt2spk_in") || die "Error: fail to open $utt2spk_in\n";
open(UO, ">$utt2spk_out") || die "Error: fail to open $utt2spk_out\n";

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
  #print STDERR $file . " " . join(" ", @utts) . "\n";
  print UO "$file $file\n";

  if (defined $text_in) {
    $text_line = "";
    foreach $utt (@utts) {
      $text_line = "$text_line " . $text{$utt}  
    }
    print TO "$file $text_line\n";
  }
}
' $data/utt2spk $dir/utt2spk $text_files

sort -u $dir/utt2spk > $dir/utt2spk.tmp
mv $dir/utt2spk.tmp $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

utils/fix_data_dir.sh $dir
