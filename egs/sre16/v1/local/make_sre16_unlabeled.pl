#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright   2017   David Snyder
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-call-my-net-training-data> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

# Handle major subset.
$out_dir_major = "$out_dir/sre16_major";
if (system("mkdir -p $out_dir_major")) {
  die "Error making directory $out_dir_major";
}

$tmp_dir_major = "$out_dir_major/tmp";
if (system("mkdir -p $tmp_dir_major") != 0) {
  die "Error making directory $tmp_dir_major";
}

open(SPKR, ">$out_dir_major/utt2spk") || die "Could not open the output file $out_dir_major/utt2spk";
open(WAV, ">$out_dir_major/wav.scp") || die "Could not open the output file $out_dir_major/wav.scp";

if (system("find $db_base/data/unlabeled/major/ -name '*.sph' > $tmp_dir_major/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_major/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$t1[0];
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";
}

close(WAV) || die;
close(SPKR) || die;

# Handle minor subset.
$out_dir_minor= "$out_dir/sre16_minor";
if (system("mkdir -p $out_dir_minor")) {
  die "Error making directory $out_dir_minor";
}

$tmp_dir_minor = "$out_dir_minor/tmp";
if (system("mkdir -p $tmp_dir_minor") != 0) {
  die "Error making directory $tmp_dir_minor";
}

open(SPKR, ">$out_dir_minor/utt2spk") || die "Could not open the output file $out_dir_minor/utt2spk";
open(WAV, ">$out_dir_minor/wav.scp") || die "Could not open the output file $out_dir_minor/wav.scp";

if (system("find $db_base/data/unlabeled/minor/ -name '*.sph' > $tmp_dir_minor/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_minor/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$t1[0];
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";
}
close(WAV) || die;
close(SPKR) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir_major/utt2spk >$out_dir_major/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_major";
}
if (system("utils/utt2spk_to_spk2utt.pl $out_dir_minor/utt2spk >$out_dir_minor/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_minor";
}
if (system("utils/fix_data_dir.sh $out_dir_major") != 0) {
  die "Error fixing data dir $out_dir_major";
}
if (system("utils/fix_data_dir.sh $out_dir_minor") != 0) {
  die "Error fixing data dir $out_dir_minor";
}
