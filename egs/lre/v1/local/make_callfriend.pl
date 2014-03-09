#! /usr/bin/perl
#
# Copyright 2014  David Snyder
# Apache 2.0.

use local::load_lang;

if (@ARGV != 5) {
  print STDERR "Usage: $0 <lang-abbreviation-file> <path-to-LDC96S*>\
                <dataset-num> <path-to-output>\n";
  print STDERR "e.g. $0 language_abbreviation.txt \
    /export/corpora5/LDC/LDC96S49 49 callfriend_lang.txt data\n";
  exit(1);
}

($lang_abbreviation_file, $db_base, $dataset,
 $lang_table, $out_base_dir) = @ARGV;

$lang = "";
open(LANGTABLE, "<$lang_table");
while($line = <LANGTABLE>) {
  chomp($line);
  ($dataset_id, $language) = split(" ", $line);
  if ($dataset eq $dataset_id) {
    $lang = $language;
    last;
  }
}

($long_lang, $abbr_lang, $num_lang) = load_lang($lang_abbreviation_file);

$callinfo_file = `find $db_base -name "callinfo.tbl"`;
open(CALLINFO, "<".$callinfo_file) || die "Cannot open $callinfo_file.";

%speaker = ();
while (<CALLINFO>) {
  ($call, $speaker) = split(' PIN=|\|');
  $speaker{$call} = $speaker;
}

foreach $set ('devtest', 'evltest', 'train') {
  $tmp_dir = "$out_base_dir/tmp";
  if (system("mkdir -p $tmp_dir") != 0) {
    die "Error making directory $tmp_dir"; 
  }
  
  if (system("find $db_base -name '*.sph' | grep '$set' > $tmp_dir/sph.list")
    != 0) {
    die "Error getting list of sph files";
  }
  
  $tmp_dir = "$out_base_dir/tmp";
  open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";
  
  while($sph = <WAVLIST>) {
    chomp($sph);
    @A = split("/", $sph);
    $basename = $A[$#A];
    $raw_basename = $basename;
    $raw_basename =~ s/\.sph$// || die "bad basename $basename";
    $wav{$raw_basename} = $sph;
  }

  close(WAVLIST) || die;
  
  $out_dir = $out_base_dir . '/ldc96s' . $dataset . '_' . $set;
  if (system("mkdir -p $out_dir") != 0) {
    die "Error making directory $out_dir"; 
  }
  
  open(WAV, ">$out_dir" . '/wav.scp') 
    || die "Failed opening output file $out_dir/wav.scp";
  open(UTT2LANG, ">$out_dir" . '/utt2lang') 
    || die "Failed opening output file $out_dir/utt2lang";
  open(UTT2SPK, ">$out_dir" . '/utt2spk') 
    || die "Failed opening output file $out_dir/utt2spk";
  
  foreach (sort keys(%wav)) {
    if (exists($speaker{$_})) {
      $spkr = $num_lang{$abbr_lang{$lang}}. "_" .$speaker{$_};
    } else {
      $spkr = $num_lang{$abbr_lang{$lang}};
    }
    $uttId = $spkr."_ldc96s".$dataset."_".$_;
    print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav{$_} |\n";
    if (exists($speaker{$_})) {
      print UTT2SPK "$uttId $spkr\n";
    } else {
      print UTT2SPK "$uttId $uttId\n";
    }
    print UTT2LANG "$uttId $lang\n";
  }
  
  close(WAV) || die;
  close(UTT2SPK) || die;
  close(UTT2LANG) || die;
  system("rm -r $out_base_dir/tmp");

  system("utils/fix_data_dir.sh $out_dir");
  (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
    || die "Error validating data dir.";
}
