#!/usr/bin/env perl
#===============================================================================
# Copyright (c) 2017  Johns Hopkins University 
#                        (Author: Jan "Yenda" Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

use strict;
use warnings;
use utf8;
use List::Util qw(max);

my $audio_width=1;
my $speaker_width=1;
my $time_width=1;

binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

if (@ARGV != 3) {
  print STDERR "$0: Error: Unsupported number of arguments: " . scalar @ARGV ."\n";
  print STDERR "  Usage: $0 <audio-files> <transripts> <destination>\n";
  print STDERR "  where\n";
  print STDERR "    <audio-files> is a file containing list of audio files\n";
  print STDERR "      (single absolute path name per line)\n";
  print STDERR "    <transcripts> is a file containing transcripts obtained\n";
  print STDERR "      obtained by processing the official SGML format\n";
  print STDERR "      transcripts. See parse_sgm.pl for further info.\n";
  print STDERR "    <destination> target directory (should already exist)\n";
  print STDERR "  See also: local/parse_sgm.pl\n";
  die;
}

my $audio_files = $ARGV[0];
my $transcripts = $ARGV[1];
my $out = $ARGV[2];

my %AUDIO;
open(my $audio_f, "<", $audio_files) 
  or die "$0: Error: Could not open $audio_files: $!\n";
while(my $line = <$audio_f>) {
  chomp $line;
  (my $basename = $line) =~ s/.*\/([^\/]+).sph/$1/g;
  $AUDIO{$basename} = $line;
}
close($audio_f);

my %TRANSCRIPT;
open(my $transcript_f, "<:encoding(utf-8)", $transcripts)
  or die "$0: Error: Could not open $transcripts: $!\n";
while(my $line = <$transcript_f>) {
  chomp $line;
  my @F = split / /, $line, 8;
  push @{$TRANSCRIPT{$F[0]}}, \@F;

  my $f1 = $F[0];
  my $f2 = $F[1];
  my $speaker = $F[2];
  my $t1 = $F[5];
  my $t2 = $F[6];

  $time_width = max $time_width, length($t1), length($t2);
  $speaker_width = max $speaker_width, length($speaker);
  $audio_width = max $audio_width, length($f1);
}
close($transcript_f);
#print Dumper(\%TRANSCRIPT);

print $time_width . " " . $speaker_width . " " . $audio_width . "\n";

my $sph2pipe = `which sph2pipe` or do {
  die "$0: Error: sph2pipe is not installed. Did you run make in the tools/ directory?\n";
};
chomp $sph2pipe;

open(my $wav_file, ">", "$out/wav.scp") 
  or die "$0: Error: Cannot create file $out/wav.scp: $!\n";
open(my $text_file, ">:encoding(utf-8)", "$out/text") 
  or die "$0: Error: Cannot create file $out/text: $!\n";
open(my $segments_file, ">", "$out/segments") 
  or die "$0: Error: Cannot create file $out/segments: $!\n";
open(my $spk_file, ">", "$out/utt2spk") 
  or die "$0: Error: Cannot create file $out/utt2spk: $!\n";

foreach my $file (sort keys %AUDIO) {
  print "$0 Error: $file does not exist in transcripts!\n"  
    unless exists $TRANSCRIPT{$file};
  my $transcripts = $TRANSCRIPT{$file};

  my $file_fmt = sprintf("%0${audio_width}s", $file);

  print $wav_file "$file_fmt $sph2pipe -f wav $AUDIO{$file}|\n";

  foreach my $utt (@{$transcripts}) {
    my $start = $utt->[5] + 0.0;  
    my $end = $utt->[6] + 0.0;
    my $start_time = sprintf("%0${time_width}d", $utt->[5]*1000);  
    my $end_time = sprintf("%0${time_width}d", $utt->[6]*1000);
    my $spk = sprintf("%0${speaker_width}s", $utt->[2]);
    my $spkid = "${file_fmt}_${spk}";
    my $uttid = "${file_fmt}_${spk}_${start_time}_${end_time}";

    print $text_file "$uttid $utt->[7]\n";
    print $spk_file "$uttid $spkid\n";
    print $segments_file "$uttid $file_fmt $start $end\n";
  }
}

close($wav_file);
close($text_file);
close($segments_file);
close($spk_file);
