#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar
# Apache 2.0.

use strict;
use File::Basename;

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2007S11> <path-to-output>\n" .
               " e.g.: $0 /export/corpora5/LDC/LDC2007S11 data\n";
  exit(1);
}

my ($db_base, $out_dir) = @ARGV;
$out_dir = "$out_dir/rt04_dev";

if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

open(SPKR, ">", "$out_dir/utt2spk") 
  or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp")
  or die "Could not open the output file $out_dir/wav.scp";

open(LIST, 'find ' . $db_base . '/data/audio/dev04s -name "*.sph" |');

while (my $line = <LIST>) {
  chomp($line);
  my ($file_id, $path, $suffix) = fileparse($line, qr/\.[^.]*/);
  if ($suffix =~ /.sph/) {
    print WAV $file_id . " sph2pipe -f wav -p -c 1 $line |\n";
  } elsif ($suffix =~ /.wav/) {
    print WAV $file_id . " $line |\n";
  } else {
    die "$0: Unknown suffix $suffix in $line\n"
  }

  print SPKR "$file_id $file_id\n";
}

close(LIST) || die;
close(WAV) || die;
close(SPKR) || die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}

system("utils/fix_data_dir.sh $out_dir");

if (system(
  "utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
