#!/usr/bin/env perl
#
# Copyright 2014  David Snyder
# Apache 2.0.

use File::Basename;

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2009S05> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2009S05 data\n";
  exit(1);
}

($db_base, $out_base_dir) = @ARGV;

$ldc_code = lc basename($db_base);

# We won't use the speaker or gender information.  Anyway it's not that useful
# as it seems to be 2-wire recordings, and everything is mixed together.

foreach $set ('arb', 'ben', 'cfr', 'rus', 'tha', 'urd', 'wuu', 'yuh') {
  $tmp_dir = "$out_base_dir/tmp";
  if (system("mkdir -p $tmp_dir") != 0) {
    die "Error making directory $tmp_dir"; 
  }
  
  if (system("find $db_base/data/lre07_tr/$set -name '*.sph' | grep '$set' > $tmp_dir/sph.list")
    != 0) {
    die "Error getting list of sph files";
  }
  
  $tmp_dir = "$out_base_dir/tmp";
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

  $out_dir = $out_base_dir . "/" . $ldc_code . '_' . $set;
  if (system("mkdir -p $out_dir") != 0) {
    die "Error making directory $out_dir"; 
  }

  open(WAV, ">$out_dir" . '/wav.scp') 
    || die "Failed opening output file $out_dir/wav.scp";
  open(UTT2LANG, ">$out_dir" . '/utt2lang') 
    || die "Failed opening output file $out_dir/utt2lang";
  open(UTT2SPK, ">$out_dir" . '/utt2spk') 
    || die "Failed opening output file $out_dir/utt2spk";

  foreach $recording (sort keys(%wav)) {
      $uttId = $ldc_code . "_" . $recording;
      print WAV "$uttId"," sph2pipe -f wav -p $wav{$recording} |\n";
      print UTT2SPK "$uttId $uttId\n";
      print UTT2LANG "$uttId $set\n";
  }

  close(WAV) || die;
  close(UTT2SPK) || die;
  close(UTT2LANG) || die;
  system("rm -r $out_base_dir/tmp");

  system("utils/fix_data_dir.sh $out_dir");
  (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
    || die "Error validating data dir.";
}
