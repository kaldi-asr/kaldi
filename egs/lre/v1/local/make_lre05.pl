#!/usr/bin/perl
# 
# Copyright 2014  David Snyder

use local::load_lang;

if (@ARGV != 3) {
  print STDERR "Usage: $0 <lang-abbreviation-file> <path-to-LDC2008S05> <output-dir>\n";
  print STDERR "e.g. $0 language_abbreviation.txt /export/corpora5/LDC/LDC2008S05: data\n";
  exit(1);
}

($lang_abbreviation_file, $db_base, $out_base_dir) = @ARGV;

($long_lang, $abbr_lang, $num_lang) = load_lang($lang_abbreviation_file);

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
open(SPK2GEN, ">$out_dir" . '/spk2gender') 
  || die "Failed opening output file $out_dir/spk2gender";

open(KEY, "<$key") 
  || die "Failed opening input file $key";

while($line = <KEY>) {
  chomp($line);
  # If the line isn't a comment
  if (index($line, "#") == -1) {
    ($fi, $lang, $conv_id, $chan, $test_cut) = split(" ", $line);
    # Verify that we have only Indian English.
    if (not ($lang eq "IE")) {
      die "$db_ie contains non-Indian English utterances.";
    }
    ($set, $part, $utt_fi) = split("/", $fi);
    ($utt, $ext) = split("[.]", $utt_fi);
    # This part of the corpus is only english.indian.
    $uttId = $num_lang{$abbr_lang{"english.indian"}}."_".$utt."_".$conv_id;
    if ($chan eq '1') {
      $uttId .= "_A";
    } else {
      $uttId .= "_B";
    }
    $wav = $db_ie . $fi;
    print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav |\n";
    print UTT2SPK "$uttId $uttId\n";
    print UTT2LANG "$uttId english.indian\n";
    # Gender info doesn't exist for the Indian English
    # only part of the corpora, defaulting to 'm.' This
    # is justified since we don't need the gender for
    # the language_id setup anyway, but it necessary for
    # combining the datasets.
    print SPK2GEN "$uttId m\n";
  }
}
close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(SPK2GEN) || die;

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

$omitted = 0;
while($line = <KEY>) {
  chomp($line);
  if (index($line, "#") == -1) {
    ($seg_id, $lang, $dialect, $conv_id, $channel, 
     $cut, $dur, $corp, $gender, $loc, $alt_lang) = split(" ", $line);
    $wav = `find $db_dir -name "$seg_id*"`;
    
    $lang = lc $lang;
    $dialect = lc $dialect;
    if ($dialect eq "na") {
      $dialect = "";
    }
    $gender = lc $gender;
    # Defaulting to male if the gender info is missing.
    if (not ($gender eq 'm' || $gender eq 'f')) {
      $gender = 'm';
    }
    
    # Elsewhere in the setup we consider Mandarin a dialect of
    # Chinese.
    if ($lang eq "mandarin") {
      %dialect_map = ("" => "mainland", "non-tw-accnt" => "mainland",
                    "non-tw-accnt,non-tw-accnt" => "mainland", 
                    "for-accnt" => "mainland", 
                    "non-tw-accnt,std" => "mainland", "std,std" => "mainland",
                    "std" => "mainland", "tw-accnt" => "taiwan");
      if (exists $dialect_map{$dialect}) {
        $dialect = $dialect_map{$dialect};
      }
      $lang = "chinese";
      $dialect = "mandarin.".$dialect;
    }

    # Hindi should be a dialect of hindustani.
    if ($lang eq "hindi") {
      $lang = "hindustani";
      if ($dialect eq "") {
        $dialect = "hindi";
      } else {
        $dialect = "hindi." . $dialect;
      }
    }

    # If the language doesn't appear in the language_abbreviation
    # file, omit it. This should only result in a few omissions.
    if (not $dialect eq "") {
      if (not (exists $abbr_lang{$lang.".".$dialect})) {
        $omitted += 1;
        next;
      }
      $lang .= ".".$dialect;
    } else {
      if (not (exists $abbr_lang{$lang})) {
        $omitted += 1;
        next;
      }
    }
    $uttId = $num_lang{$abbr_lang{$lang}}."_".$seg_id."_".$conv_id."_".$channel;

    print WAV "$uttId"," sph2pipe -f wav -p -c 1 $wav |\n";
    print UTT2SPK "$uttId $uttId\n";
    print SPK2GEN "$uttId $gender\n";
    print UTT2LANG "$uttId $lang\n";
  }
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
close(SPK2GEN) || die;

print "Omitted $omitted utterances because they don't appear ",
      "in language_abbreviations.txt\n";

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";
