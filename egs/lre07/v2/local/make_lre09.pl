#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright 2014  David Snyder

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LRE2009/eval> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora5/NIST/LRE/LRE2009/eval data\n";
  exit(1);
}

($base, $out_base_dir) = @ARGV;

$out_dir = $out_base_dir . "/lre09";
$tmp_dir = "$out_dir/tmp";
$db_file = $base . "/keys/NIST_LRE09_segment.key.v0.txt";
open(DB, "<$db_file")
  || die "Failed opening input file $db_file";

if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $out_dir"; 
}

open(WAV, ">$out_dir" . '/wav.scp') 
  || die "Failed opening output file $out_dir/wav.scp";
open(UTT2LANG, ">$out_dir" . '/utt2lang') 
  || die "Failed opening output file $out_dir/utt2lang";
open(UTT2SPK, ">$out_dir" . '/utt2spk') 
  || die "Failed opening output file $out_dir/utt2spk";


if (system("find $base -name '*.sph' > $tmp_dir/sph.list")
  != 0) {
  die "Error getting list of sph files";
}
  
open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";

%wav = ();
while($sph = <WAVLIST>) {
  chomp($sph);
  @A = split("/", $sph);
  $basename = $A[$#A];
  $raw_basename = $basename;
  $raw_basename =~ s/\.sph$// || die "bad basename $basename";
  $wav{$raw_basename} = $sph;
}

close(WAVLIST) || die;

while($line = <DB>) {
  chomp($line);

  if (index($line, "#") != -1) {
    next;
  }
  @toks = split(",", $line);
  $seg_id = lc $toks[0];
  $lang = lc $toks[1];
  $channel = $toks[5];
  if ($channel eq "X") {
    next;
  } elsif ($channel eq "a") {
    $channel = "1";
  } elsif ($channel eq "b") {
    $channel = "2";
  } else {
    die "Invalid channel $channel";
  }
  
  if (! -f $wav{$seg_id}) {
    print STDERR "No such file $wav{$seg_id}\n";
    next;
  }

  $uttId = "lre09_${seg_id}_${channel}";
  
  print WAV "$uttId"," sph2pipe -f wav -p -c ${channel} $wav{$seg_id} |\n";
  print UTT2SPK "$uttId $uttId\n";
  print UTT2LANG "$uttId $lang\n";
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(DB) || die;
system("rm -r $tmp_dir");

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";
