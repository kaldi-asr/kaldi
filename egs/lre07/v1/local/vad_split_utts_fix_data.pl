#! /usr/bin/perl
#
# Copyright 2014  David Snyder
# Apache 2.0.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <in-data-dir> <out-data-dir>\n";
  print STDERR "e.g. $0 data/train_unsplit data/train\n";
  exit(1);
}

($in_dir, $out_dir) = @ARGV;

%utt2lang = ();
%utt2spk = ();

open(UTT2LANG, "<$in_dir/utt2lang") or die "Cannot open utt2lang";
while($line = <UTT2LANG>) {
  ($utt, $lang) = split(" ", $line);
  $utt2lang{$utt} = $lang;
}
close(UTT2LANG) or die;

open(UTT2SPK, "<$in_dir/utt2spk") or die "Cannot open utt2spk";
while($line = <UTT2SPK>) {
  ($utt, $spk) = split(" ", $line);
  $utt2spk{$utt} = $spk;
}
close(UTT2SPK) or die;

open(FEATSEG, "<$out_dir/frame_indexed_segments") 
  or die "Unable to open feats_segment";
open(UTT2LANG, ">$out_dir/utt2lang") or die "Cannot open utt2lang";
open(UTT2SPK, ">$out_dir/utt2spk") or die "Cannot open utt2spk";
open(SEGMENT, ">$out_dir/segments") or die "Cannot open segments";

while($seg = <FEATSEG>) {
  ($split_utt, $utt, $start, $end) = split(" ", $seg);
  print UTT2LANG "$split_utt $utt2lang{$utt}\n";
  print UTT2SPK "$split_utt $utt\n";
  $start_t = $start * 0.01;
  $end_t = $end * 0.01;
  print SEGMENT "$split_utt $utt $start_t $end_t\n";
}

close(FEATSEG) || die;
close(UTT2LANG) || die;
close(UTT2SPK) || die;
close(SEGMENT) || die;
system("utils/fix_data_dir.sh $out_dir");
