#!/usr/bin/perl
# 
# Copyright 2014  David Snyder

use local::load_lang;

if (@ARGV != 3) {
  print STDERR "Usage: $0 <lang-abbreviation-file> <path-to-LDC2006S31>\
     <output-dir>\n";
  print STDERR "e.g. $0 language_abbreviation.txt \
     /export/corpora4/LDC/LDC2006S31 data\n";
  exit(1);
}

($lang_abbreviation_file, $db_base, $out_base_dir) = @ARGV;

($long_lang, $abbr_lang, $num_lang) = load_lang($lang_abbreviation_file);


$lang_file = $db_base . "/docs/LID03_KEY.v3";
open(LANGS, "<$lang_file") 
  || die "Failed opening input file $lang_file";


$omitted = 0;
$out_dir = $out_base_dir . "/lre03";
$db_dir = $db_base . "/data/lid03e1";
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

while($line = <LANGS>) {
  chomp($line);
  if (index($line, ";") == -1) {
    @toks = split(" ", $line);
    $seg_id = lc $toks[0];
    $lang = lc $toks[1];
    $conv_id = $toks[2];
    $channel = $toks[3];
    $gender = lc $toks[6];
    $channel = substr($channel, 0, 1);

    $wav = `find $db_dir -name "$seg_id*"`;
   
    # Small adjustments needed to language format. 
    if ($lang eq "mandarin") {
      $lang = "chinese.mandarin.mainland";
    }
    if ($lang eq "english") {
      $lang = "english.varied";
    }
    if ($lang eq "arabic") {
      $lang = "arabic.standard";
    }
    if ($lang eq "hindi") {
      $lang = "hindustani.hindi";
    }
    if (not exists $abbr_lang{$lang}) {
      print "Not including $lang\n";
      $omitted += 1;
      next;
    }
    $uttId = $num_lang{$abbr_lang{$lang}}."_".$seg_id."_".$conv_id."_".$channel;

    print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav"." |\n";
    print UTT2SPK "$uttId $uttId\n";
    print UTT2LANG "$uttId $lang\n";
    print SPK2GEN "$uttId $gender\n";
  }
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(SPK2GEN) || die;
close(LANGS) || die;

print "Omitted $omitted utterances because they don't appear ",
      "in language_abbreviation.txt\n";

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";

%lid96_lang_map = ();
$lid96de_lang_file = "local/lid96de_lang.txt";
open(LID96_LANG, "<$lid96de_lang_file") 
  || die "Unable to open for input $lid96de_lang_file";
while($line = <LID96_LANG>) {
  chomp($line);
  ($lid96, $lang) = split(" ", $line);
  $lid96_lang_map{$lid96} = $lang;
}
close(LID96_LANG) || die;

for $set ("lid96d1", "lid96e1") {
  $out_dir = $out_base_dir . "/$set";
  $db_dir = $db_base . "/data/$set/test/";
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
  for $duration ("10", "3", "30") {
    $key = "$db_dir/$duration/seg_lang.ndx";
    open(KEY, "<$key") 
      || die "Failed opening input file $key";
    while($line = <KEY>) {
      chomp($line);
      ($seg_id, $lang) = split(" ", $line);
      
      if (not exists $abbr_lang{$lang}) {
        if (not exists $lid96_lang_map{$lang}) {
          $omitted += 1;
          next;
        }
        $lang = $lid96_lang_map{$lang};
      }

      $wav = "$db_dir/$duration/$seg_id.sph";
      $uttId = $num_lang{$abbr_lang{$lang}}."_".$seg_id."_".$set."_".$duration;

      print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav"." |\n";
      print UTT2SPK "$uttId $uttId\n";
      print UTT2LANG "$uttId $lang\n";
      # Gender information is absent here, defaulting to male
      # since the language_id system doesn't take gender into account,
      # currently.
      print SPK2GEN "$uttId m\n";
    }
  }
  close(KEY) || die;
  close(WAV) || die;
  close(UTT2SPK) || die;
  close(SPK2GEN) || die;
  close(UTT2LANG) || die;
  print "Omitted $omitted utterances because they don't appear ",
      "in language_abbreviation.txt\n";

  system("utils/fix_data_dir.sh $out_dir");
  (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
     || die "Error validating data dir.";
}
