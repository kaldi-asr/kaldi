#! /usr/bin/perl

use strict;
use warnings;
use local::load_lang;

my ($dataset, $in_top, $out_top) = @ARGV;
die 'Usage: ' . File::Basename::basename($0)
    . " {48|49|54|55|56|57|58} in-dir out-dir\n"
    unless @ARGV == 3 && $dataset =~ /^4[89]|5[4-8]$/o
	   && $in_top && $out_top;

sub open_or_die ($$) {
    my ($mode, $path, $file) = @_;
    open($file, $mode, $path) or die "$path: $!\n";
    return ($path, $file);
}
my $lang_abbreviation_file = "local/language_abbreviations.txt";
my ($long_lang, $abbr_lang, $num_lang) = load_lang($lang_abbreviation_file);
my %doc = (
    '48' => '/callfriend_fre_1/cf_fre/docs/',
    '49' => '/doc/',
    '54' => '/doc/',
    '55' => '/doc/',
    '56' => '/doc/',
    '57' => '/docs/',
    '58' => '/doc/'
);

my $doc = $in_top . $doc{$dataset};
my ($meta_path, $meta_file, %speaker) =
    open_or_die('<', $doc . 'callinfo.tbl');

while (<$meta_file>) {
    my ($call, $speaker) = split(' PIN=|\|');
    $speaker{$call} = $speaker;
}

close $meta_file or warn "$meta_path: $!\n";
($meta_path, $meta_file) = open_or_die('<', $doc . 'spkrinfo.tbl');
my %gender;

while (<$meta_file>) {
    my ($call, $gender) = split(',');
    $gender =~ tr/FM/fm/;
    $gender{$call} = $gender;
}

close $meta_file or warn "$meta_path: $!\n";
($, , $\) = (' ', "\n");
$out_top .= '/ldc96s' . $dataset . '_';

my %data = (
    '48' => '/callfriend_fre_1/cf_fre/data/',
    '49' => '/data/',
    '54' => '/data/',
    '55' => '/data/',
    '56' => '/data/',
    '57' => '/data/',
    '58' => '/cf_spa_n/'
);
my %lang_name = (
    '48' => 'french',
    '49' => 'arabic.standard',
    '54' => 'korean',
    '55' => 'chinese.mandarin.mainland',
    '56' => 'chinese.mandarin.taiwan',
    '57' => 'spanish.caribbean',
    '58' => 'spanish.noncaribbean'
);
my $lang_code = $::num_lang{$::abbr_lang{$lang_name{$dataset}}};
$in_top .= $data{$dataset};
my $lang_name = $lang_name{$dataset};

sub open4sort ($;$) {
    my ($path, $flags) = @_;
    open_or_die('|-',
		($flags ? 'sort ' . $flags . ' >' : 'sort >')
		. $path);
}

use File::Path;
use File::Find;

foreach ('devtest', 'evltest', 'train') {

    my $out_sub = $out_top . $_;
    File::Path::make_path($out_sub);
    $out_sub .= '/';

    my ($wav_path, $wav_file) = open4sort($out_sub . 'wav.scp');
    my ($utt2lang_path, $utt2lang_file) =
	open4sort($out_sub . 'utt2lang');
    my ($utt2spk_path, $utt2spk_file) =
	open4sort($out_sub . 'utt2spk');
    my ($spk2gender_path, $spk2gender_file) =
	open4sort($out_sub . 'spk2gender', '-u');

    File::Find::find(sub {

	my ($call   ) = /^(.*)\.sph$/o or return;
	my  $speaker  = $speaker{$call};

	if (!$speaker) {
	    warn "$call: No call metadata.\n";
	    return;
	}

	my $utt = $lang_code . '_' . $speaker . '_ldc96s' . $dataset
		  . '_' . $call;
	print $wav_file
	    $utt, 'sph2pipe -f wav -p -c 1', $File::Find::name;
	print $utt2lang_file   $utt    , $lang_name;
	print $utt2spk_file    $utt    , $lang_code . "_" . $speaker;
	print $spk2gender_file $speaker, $gender{$call};

    }, $in_top . $_);

    close $wav_file        or warn "$wav_path: $!\n";
    close $utt2lang_file   or warn "$utt2lang_path: $!\n";
    close $utt2spk_file    or warn "$utt2spk_path: $!\n";
    close $spk2gender_file or warn "$spk2gender_path: $!\n";

    print("utils/utt2spk_to_spk2utt.pl ${out_sub}utt2spk > ${out_sub}spk2utt");
    if (system("utils/utt2spk_to_spk2utt.pl ${out_sub}utt2spk > ${out_sub}spk2utt") != 0) {
	die "${out_sub}utt2spk: utt2spk_to_spk2utt.pl: $!\n";
    }
    if (system("utils/utt2spk_to_spk2utt.pl ${out_sub}utt2lang > ${out_sub}lang2utt") != 0) {
	die "${out_sub}utt2lang: utt2spk_to_spk2utt.pl: $!\n";
    }
    system("utils/fix_data_dir.sh $out_sub 1");
    system("utils/validate_data_dir.sh --no-text --no-feats $out_sub");

}
