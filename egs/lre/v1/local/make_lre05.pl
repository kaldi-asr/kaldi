#!/usr/bin/perl
#
# Copyright 2014  David Snyder

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2008S05> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2008S05 data\n";
  exit(1);
}

($db_base, $out_base_dir) = @ARGV;


# This is the Indian English part of the corpora, we will prep it first.
$db_ie =  $db_base . "/data/lid05d1/";
$key = $db_ie . "key.txt";
$out_dir = $out_base_dir . "/lid05d1/";
if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir"; 
}

open(WAV, ">$out_dir" . '/wav.scp') 
  || die "Failed opening output file $out_dir/wav.scp";
open(UTT2LANG, ">$out_dir" . '/utt2lang') 
  || die "Failed opening output file $out_dir/utt2lang";
open(UTT2SPK, ">$out_dir" . '/utt2spk') 
  || die "Failed opening output file $out_dir/utt2spk";

open(KEY, "<$key") 
  || die "Failed opening input file $key";

while($line = <KEY>) {
  chomp($line);
  # If the line isn't a comment
  if (index($line, "#") == -1) {
    ($fi, $lang, $conv_id, $channel, $test_cut) = split(" ", $line);
    # Verify that we have only Indian English.
    if (not ($lang eq "IE")) {
      die "$db_ie contains non-Indian English utterances.";
    }
    ($set, $part, $utt_fi) = split("/", $fi);
    ($utt, $ext) = split("[.]", $utt_fi);
    # This part of the corpus is only english.indian.
    $uttId = "lid05d1_$utt";
    $wav = $db_ie . $fi;
    if (! -f $wav) {
      print STDERR "No such file $wav (skipping)\n";
      next;
    }
    $channel =~ tr/AB/12/;
    print WAV "$uttId"," sph2pipe -f wav -p -c $channel $wav |\n";
    print UTT2SPK "$uttId $uttId\n";
    print UTT2LANG "$uttId english.indian\n";
  }
}
close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";

$out_dir = $out_base_dir . "/lid05e1/";
$db_dir = $db_base . "/data/lid05e1/";

$key = $db_dir . "lid05e_key_v2.txt";
open(KEY, "<$key") 
  || die "Failed opening input file $key";

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

while($line = <KEY>) {
  chomp($line);
  if (index($line, "#") == -1) {
    ($seg_id, $lang, $dialect, $conv_id, $channel,
     $cut, $dur, $corp, $gender, $loc, $alt_lang) = split(" ", $line);
    $wav = "$db_dir/test/${dur}/${seg_id}.sph";
    if (! -f $wav) {
      print STDERR "No such file $wav (skipping this utterance)\n";
      next;
    }
    $lang = lc $lang;
    $dialect = lc $dialect;
    if ($dialect ne "na") {
      $full_lang = "$lang.$dialect";
    } else {
      $full_lang = $lang;
    }

    $gender = lc $gender;
    # Defaulting to male if the gender info is missing.
    if (not ($gender eq 'm' || $gender eq 'f')) {
      $gender = 'm';
    }

    $uttId = "lid05e1_".$seg_id;
    $channel =~ tr/AB/12/;

    print WAV "$uttId"," sph2pipe -f wav -p -c ${channel} $wav |\n";
    print UTT2SPK "$uttId $uttId\n";
    print SPK2GEN "$uttId $gender\n";
    print UTT2LANG "$uttId $full_lang\n";
  }
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(SPK2GEN) || die;

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";
