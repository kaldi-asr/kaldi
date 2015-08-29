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
  utils/copy_data_dir.sh $data $dir
  exit 0
fi

mkdir -p $dir
cp $data/wav.scp $dir
cp $data/reco2file_and_channel $dir

cat $data/segments | perl -e '
($text_in, $utt2spk_in, $text_out, $utt2spk_out) = @ARGV;
open(TI, "<$text_in") || die "Error: fail to open $text_in\n";
open(TO, ">$text_out") || die "Error: fail to open $text_out\n";
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

while (<TI>) {
  chomp;
  my @col = split;
  @col >= 2 or die "bad line $_\n";

  my $utt = shift @col;
  $text{$utt} = join(" ", @col);
}
while (<UI>) {
  chomp; 
  my @col = split;
  $utt2spk{$col[0]} = $col[1];
}

foreach $file (keys %file2utt) {
  my @utts = @{$file2utt{$file}};
  #print STDERR $file . " " . join(" ", @utts) . "\n";
  $text_line = "";
  foreach $utt (@utts) {
    defined $text{$utt} or die "Unknown utterance $utt in text\n";
    defined $utt2spk{$utt} or die "Unknown utterance $utt in utt2spk\n";

    $text_line .=  " " . $text{$utt};
    print UO "$file $utt2spk{$utt}\n";
  }
  print TO "$file $text_line\n";
}
' $data/text $data/utt2spk $dir/text $dir/utt2spk

sort -u $dir/utt2spk > $dir/utt2spk.tmp
mv $dir/utt2spk.tmp $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

utils/fix_data_dir.sh $dir
