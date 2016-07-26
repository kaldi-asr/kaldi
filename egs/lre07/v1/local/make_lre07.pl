#!/usr/bin/env perl
#
# Copyright 2014  David Snyder
# Usage: make_lre07.pl <path-to-LDC2009S04> <output-dir>


if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2009S04> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2009S04 data/lre07\n";
  exit(1);
}

($db_base, $dir) = @ARGV;

$tmp_dir = "$dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";

while (<WAVLIST>) {
  chomp;
  $sph = $_;
  @A = split("/", $sph);
  $basename = $A[$#A];
  $raw_basename = $basename;
  $raw_basename =~ s/\.sph$// || die "bad basename $basename";
  $wav{$raw_basename} = $sph;
}
open(WAV, ">$dir/wav.scp") || die "Failed opening output file $out_dir/wav.scp";
open(UTT2SPK, ">$dir/utt2spk") || die "Failed opening output file $dir/utt2spk";
open(SPK2UTT, ">$dir/spk2utt") || die "Failed opening output file $dir/spk2utt";
open(UTT2LANG, ">$dir/utt2lang") || die "Failed opening output file $dir/utt2lang";
open(DUR3, ">$dir/3sec") || die "Failed opening output file $dir/3sec";
open(DUR10, ">$dir/10sec") || die "Failed opening output file $dir/10sec";
open(DUR30, ">$dir/30sec") || die "Failed opening output file $dir/30sec";

my $key_str = `wget -qO- "http://www.openslr.org/resources/23/lre07_key.txt"`;
@key_lines = split("\n",$key_str);
%utt2lang = ();
%utt2dur = ();
foreach (@key_lines) {
  @words = split(' ', $_);
  if (index($words[0], "#") == -1) {
    $utt2lang{$words[0]} = $words[1];
    $utt2dur{$words[0]} = $words[5];
  }
}

foreach (sort keys(%wav)) {
  $uttId = $_;
  print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav{$uttId} |\n";
  # We don't really have speaker info, so just make it the same as the
  # utterances: an identity map.
  print UTT2SPK "$uttId $uttId\n";
  print SPK2UTT "$uttId $uttId\n";
  print UTT2LANG "$uttId $utt2lang{$uttId}\n";
  if ($utt2dur{$uttId} == 3) {
    print DUR3 "$uttId\n";
  } elsif ($utt2dur{$uttId} == 10) {
    print DUR10 "$uttId\n";
  } elsif ($utt2dur{$uttId} == 30) {
    print DUR30 "$uttId\n";
  } else {
    die "Invalid nominal duration in test segment";
  }
}
close(WAV) || die;
close(UTT2SPK) || die;
close(SPK2UTT) || die;
close(UTT2LANG) || die;
close(DUR3) || die;
close(DUR10) || die;
close(DUR30) || die;
close(WAVLIST) || die;
system("rm -r $dir/tmp");

(system("utils/validate_data_dir.sh --no-text --no-feats $dir") == 0) || die "Error validating data dir.";
