#!/usr/bin/env perl
# Copyright 2014  Gaurav Kumar.   Apache 2.0

use utf8;
use File::Basename;

($tmpdir)=@ARGV;
$trans="$tmpdir/callhome_train_transcripts.flist";
$reco="$tmpdir/callhome_reco2file_and_channel";
open(T, "<", "$trans") || die "Can't open transcripts file";
open(R, "|sort >$reco") || die "Can't open reco2file_and_channel file $!";
open(O, ">$tmpdir/callhome.text.1") || die "Can't open text file for writing";
open(G, ">$tmpdir/callhome_spk2gendertmp") || die "Can't open the speaker to gender map file";
binmode(O, ":utf8");
while (<T>) {
  $file = $_;
  m:([^/]+)\.txt: || die "Bad filename $_";
  $call_id = $1;
  print R "$call_id-A $call_id A\n";
  print R "$call_id-B $call_id B\n";
  open(I, "<$file") || die "Opening file $_";
  binmode(I, ":iso88591");
  #Now read each line and extract information
  while (<I>) {
        #136.37 138.10 B: Ah, bueno, mamita.
    chomp;

    my @stringComponents = split(":", $_, 2);
          my @timeInfo = split(" ", $stringComponents[0]);
          $stringComponents[1] =~ s/^\s+|\s+$//g ;
          my $words = $stringComponents[1];
    #Check number of components in this array
    if ((scalar @stringComponents) >= 2) {
      $start = sprintf("%06d", $timeInfo[0] * 100);
      $end = sprintf("%06d", $timeInfo[1] * 100);
      length($end) > 6 && die "Time too long $end in $file";
      $side = "A";
      if (index($timeInfo[2], "B") != -1) {
        $side = "B";
      }
      $utt_id = "${call_id}-$side-$start-$end";
      $speaker_id = "${call_id}-$side";
      # All speakers are treated as male because speaker gender info
      # is missing in this file
      $gender = "m";
      print G "$speaker_id $gender\n" || die "Error writing to speaker2gender file";
                        $words =~ s|\[\[[^]]*\]\]||g;    #removes comments
                        $words =~ s|\{laugh\}|\$laughter\$|g;    # replaces laughter tmp
                        $words =~ s|\[laugh\]|\$laughter\$|g;    # replaces laughter tmp
                        $words =~ s|\{[^}]*\}|\[noise\]|g;       # replaces noise
                        $words =~ s|\[[^]]*\]|\[noise\]|g;       # replaces noise
                        $words =~ s|\[/*([^]]*)\]|\[noise\]|g;   # replaces end of noise
                        $words =~ s|\$laughter\$|\[laughter\]|g; # replaces laughter again
                        $words =~ s|\(\(([^)]*)\)\)|\1|g;        # replaces unintelligible speech
                        $words =~ s|<\?([^>]*)>|\1|g;            # for unrecognized language
                        $words =~ s|background speech|\[noise\]|g;
                        $words =~ s|background noise|\[noise\]|g;
                        $words =~ s/\[/larrow/g;
                        $words =~ s/\]/rarrow/g;
                        $words =~ s/[[:punct:]]//g;
                        $words =~ s/larrow/\[/g;
                        $words =~ s/rarrow/\]/g;
      $words =~ s/[¿¡]//g;
                        $words =~ s/\h+/ /g;                     # horizontal whitespace characters
      $words = lc($words);
      print O "$utt_id $words\n" || die "Error writing to text file";
    }
  }
  close(I);
}
close(T);
close(R);
close(O);
close(G);
