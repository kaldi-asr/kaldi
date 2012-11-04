#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage: utils/write_kwslist.pl [options] <raw_result_in|-> <kwslist_out|->
 e.g.: utils/write_kwslist.pl --flen=0.01 --duration=1000 --segments=data/eval/segments
                              --normalize=true --map-utter=data/kws/utter_map raw_results kwslist.xml

Allowed options:
  --beta                      : Beta value when computing ATWV              (float,   default = 999.9)
  --duration                  : Duration of all audio, you must set this    (float,   default = 999.9)
  --ecf-filename              : ECF file name                               (string,  default = "") 
  --flen                      : Frame length                                (float,   default = 0.01)
  --index-size                : Size of index                               (float,   default = 0)
  --language                  : Language type                               (string,  default = "cantonese")
  --map-utter                 : Map utterance for evaluation                (string,  default = "")
  --normalize                 : Normalize scores or not                     (boolean, default = false)
  --segments                  : Segments file from Kaldi                    (string,  default = "")
  --system-id                 : System ID                                   (string,  default = "")
EOU

my $segment = "";
my $flen = 0.01;
my $beta = 999.9;
my $duration = 999.9;
my $language = "cantonese";
my $ecf_filename = "";
my $index_size = 0;
my $system_id = "";
my $normalize = "false";
my $map_utter = "";
GetOptions('segments=s'     => \$segment,
           'flen=f'         => \$flen,
           'beta=f'         => \$beta,
           'duration=f'     => \$duration,
           'language=s'     => \$language,
           'ecf-filename=s' => \$ecf_filename,
           'index-size=f'   => \$index_size,
           'system-id=s'    => \$system_id,
           'normalize=s'    => \$normalize,
           'map-utter=s'    => \$map_utter); 

if ($normalize ne "true" && $normalize ne "false") {
  die "Bad value for option --normalize. \n";
}

if ($segment) {
  if (!open(SEG, "<$segment")) {print "Fail to open segment file: $segment\n"; exit 1;}
}

if ($map_utter) {
  if (!open(UTT, "<$map_utter")) {print "Fail to open utterance table: $map_utter\n"; exit 1;}
}

if(@ARGV != 2) {
  die $Usage;
}

# Get parameters
my $filein = shift @ARGV;
my $fileout = shift @ARGV;

# Get input source
my $source = "";
if ($filein eq "-") {
  $source = "STDIN";
} else {
  if (!open(I, "<$filein")) {print "Fail to open input file: $filein\n"; exit 1;}
  $source = "I";
}

# Open output fst list
my $sourceout = "";
if ($fileout ne "-") {
  if (!open(O, ">$fileout")) {print "Fail to open output file: $fileout\n"; exit 1;}
  $sourceout = "O";
}

# Get symbol table and start time
my %tbeg = ();
if ($segment) {
  while(<SEG>) {
    chomp;
    my @col = split(" ", $_);
    @col == 4 || die "Bad number of columns in $segment\n";
    $tbeg{$col[0]} = $col[2];
  }
}

# Get utterance mapper
my %utter_mapper = ();
if ($map_utter) {
  while(<UTT>) {
    chomp;
    my @col = split(" ", $_);
    @col == 2 || die "Bad number of columns in $map_utter\n";
    $utter_mapper{$col[0]} = $col[1];
  }
}

# Processing
my %results = ();
while (<$source>) {
  chomp;
  my @col = split(" ", $_);
  @col == 5 || die "Bad number of columns in raw results\n";
  my $term_id = shift @col;
  my $utt = $col[0];
  my $start = $col[1]*$flen;
  my $dur = $col[2]*$flen-$start;
  my $score = exp(-$col[3]);

  if ($segment) {
    $start += $tbeg{$utt};
  }

  push(@{$results{$term_id}}, [$utt, $start, $dur, $score]);
}

my $key;
my $iterm;
my %Ntrue = ();
foreach $key (keys %results) {
  foreach $iterm (@{$results{$key}}) {
    if (!defined($Ntrue{$key})) {
      $Ntrue{$key} = 0.0;
    }
    $Ntrue{$key} += @{$iterm}[3];
  }
}

eval "print $sourceout \'<kwslist kwlist_filename=\"$ecf_filename\" language=\"$language\" system_id=\"$system_id\">\n\'";
foreach $key (sort {($a =~ /([0-9]*)$/)[0] <=> ($b =~ /([0-9]*)$/)[0]} (keys %results)) {
  my $term_search_time = "1";
  my $oov_term_count = "0";
  $key =~ m/-([0-9]*)$/;
  my $suffix = sprintf("%04d", $1);
  eval "print $sourceout \'<detected_kwlist kwid=\"$key\" search_time=\"$term_search_time\" oov_count=\"$oov_term_count\">\n\'";
  # Collect results
  my %list = ();
  my @list = ();
  foreach $iterm (@{$results{$key}}) {
    my $decision = "NO";
    my $bias = 0.0;
    my $score = ($Ntrue{$key}+$bias)/($duration/$beta+($beta-1)/$beta*($Ntrue{$key}+$bias)); 
    if (@{$iterm}[3] > $score) {
      # if (@{$iterm}[3] > $score && @{$iterm}[2] > 0.05 && @{$iterm}[2] < 2) {
      $decision = "YES";
    }
    if ($normalize eq "true") {
      $score = (@{$iterm}[3]-$score+1)/2;             # Normalize here
    } else {
      $score = @{$iterm}[3];
    }
    @{$iterm}[1] = sprintf("%.2f", @{$iterm}[1]);
    @{$iterm}[2] = sprintf("%.2f", @{$iterm}[2]);
    $score = sprintf("%.2f", $score);
    my $utter = @{$iterm}[0];
    if ($map_utter) {
      $utter = $utter_mapper{$utter};
    }
    push (@list, "<kw file=\"$utter\" channel=\"1\" tbeg=\"@{$iterm}[1]\" dur=\"@{$iterm}[2]\" score=\"$score\" decision=\"$decision\"/>\n");
    $list{$score} = 1;
  }
  # Now sort results by score
  foreach(sort {$b <=> $a} keys %list) {
    my $scorekey = $_;
    foreach(grep(/score=\"$scorekey\"/, @list)) {
      eval "print $sourceout \'$_\'";
    }
  }
  eval "print $sourceout \'</detected_kwlist>\n\'";
}
eval "print $sourceout \'</kwslist>\n\'";

if ($segment) {close(SEG);}
if ($filein  ne "-") {close(I);}
if ($fileout ne "-") {close(O);}
