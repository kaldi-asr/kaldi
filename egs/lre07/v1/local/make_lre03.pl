#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright 2014  David Snyder  Daniel Povey
 


if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2006S31> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora4/LDC/LDC2006S31 data\n";
  exit(1);
}

($base, $out_base_dir) = @ARGV;

$db_file = $base . "/docs/LID03_KEY.v3";
open(DB, "<$db_file")
  || die "Failed opening input file $db_file";

$out_dir = $out_base_dir . "/lre03";
$data_dir = $base . "/data/lid03e1";
if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir"; 
}

open(WAV, ">$out_dir" . '/wav.scp') 
  || die "Failed opening output file $out_dir/wav.scp";
open(UTT2LANG, ">$out_dir" . '/utt2lang') 
  || die "Failed opening output file $out_dir/utt2lang";
open(UTT2SPK, ">$out_dir" . '/utt2spk') 
  || die "Failed opening output file $out_dir/utt2spk";
open(SPK2GEN, ">$out_dir" . '/spk2gender')
  || die "Failed opening output file $out_dir/spk2gender";

while($line = <DB>) {
  chomp($line);
  @toks = split(" ", $line);
  $seg_id = lc $toks[0];
  $lang = lc $toks[1];
  # $conv_id = $toks[2];
  $channel = $toks[3];
  $duration = $toks[4];
  $gender = lc $toks[6];
  $channel = substr($channel, 1, 1); # they are either A1 or B2: we want the
                                     # numeric channel.

  $wav = "$base/data/lid03e1/test/$duration/$seg_id.sph";
  if (! -f $wav) {
    print STDERR "No such file $wav\n";
    next;
  }
  $uttId = "lre03_${seg_id}";
  
  print WAV "$uttId"," sph2pipe -f wav -p -c ${channel} $wav |\n";
  print UTT2SPK "$uttId $uttId\n";
  print UTT2LANG "$uttId $lang\n";
  print SPK2GEN "$uttId $gender\n";
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(SPK2GEN) || die;
close(DB) || die;

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";


for $set ("lid96d1", "lid96e1") {
  $out_dir = $out_base_dir . "/$set";
  $data_dir = $base . "/data/$set/test/";
  if (system("mkdir -p $out_dir") != 0) {
    die "Error making directory $out_dir"; 
  }
  
  open(WAV, ">$out_dir" . '/wav.scp') 
    || die "Failed opening output file $out_dir/wav.scp";
  open(UTT2LANG, ">$out_dir" . '/utt2lang') 
    || die "Failed opening output file $out_dir/utt2lang";
  open(UTT2SPK, ">$out_dir" . '/utt2spk') 
    || die "Failed opening output file $out_dir/utt2spk";
  for $duration ("10", "3", "30") {
    $key = "$data_dir/$duration/seg_lang.ndx";
    open(KEY, "<$key") 
      || die "Failed opening input file $key";
    while ($line = <KEY>) {
      chomp($line);
      ($seg_id, $lang) = split(" ", $line);

      $wav = "$data_dir/$duration/$seg_id.sph";
      $uttId = "${set}_${seg_id}";
      print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav |\n";
      print UTT2SPK "$uttId $uttId\n";
      print UTT2LANG "$uttId $lang\n";
      # Gender information is absent here, not outputting spk2gender file.
    }
    close(KEY) || die;
  }
  close(WAV) || die;
  close(UTT2SPK) || die;
  close(UTT2LANG) || die;
  system("utils/fix_data_dir.sh $out_dir");
  (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
    || die "Error validating data dir.";
}
