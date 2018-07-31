#!/usr/bin/env perl
#===============================================================================
# Copyright 2015  (Author: Yenda Trmal <jtrmal@gmail.com>)
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

my $Usage = <<EOU;
Writes the results file (native kaldi-kws format) into kwslist.xml (F4DE/NIST)
format.

Usage:
  cat results | $0 --flen 0.01 > kwslist.xml

Allowed options:
  --flen           : duration (in seconds) of audio/feature frame
  --language       : language (string, default "")
  --kwlist-id      : kwlist.xml name (string, default "")
  --system-id      : name of the system (string, default "")
  --digits         : how many digits should the scores be rounded to?
                     (int, default 2). Sometimes F4DE gets extremely slow
                     when the scores have too many digits (perhaps some sweping
                     issue). This switch can be used to prevent it.
EOU

use strict;
use warnings;
use utf8;

use POSIX;
use Data::Dumper;
use Getopt::Long;

my $flen = 0.01;
my $language="";
my $kwlist_filename="";
my $system_id="";
my $digits = 2;

GetOptions("flen=f"      => \$flen,
           "language=s"  => \$language,
           "kwlist-id=s" => \$kwlist_filename,
           "system-id=s" => \$system_id,
           "digits=i"    => \$digits) or do {
  print STDERR "Cannot parse the command-line options.\n";
  print STDERR "$Usage\n";
  die "Cannot continue.\n";
};

if (@ARGV != 0) {
  print STDERR "Incorrect number of command-line arguments\n";
  print STDERR "$Usage\n";
  die "Cannot continue.\n";
}

sub KwsOutputSort {
  my $a = shift @_;
  my $b = shift @_;

  if ($a->[4] != $b->[4]) {
    #score
    return $b->[4] <=> $a->[4];
  } elsif ($a->[1] ne $b->[1]) {
    return $a->[1] cmp $b->[1];
  } else {
    return  $a->[2] <=> $b->[2];
  }
}

sub PrettyPrint {
  my @instances = sort {KwsOutputSort($a, $b)} @{shift @_};

  return if @instances <= 0;
  my $kwid=$instances[0]->[0];

  print "  <detected_kwlist kwid=\"$kwid\" search_time=\"1\" oov_count=\"0\">\n";
  foreach my $elem(@instances) {
    (my $kwidx, my $file, my $start, my $end, my $score) = @{$elem};
    my $filename="file=\"$file\"";

    # this is because the decision has to be done on the already
    # rounded number (otherwise it can confuse F4DE.
    # It's because we do the decision based on the non-rounded score
    # but F4DE will see only the rounded score, so the decision
    # won't be correctly aligned with the score (especially, for
    # some numbers with score 0.5 the decision will be "YES" and for
    # other with the same score, the decision will be "NO"
    $score = sprintf "%.${digits}f", $score;
    my $decision=$score >= 0.5 ? "decision=\"YES\"" : "decision=\"NO\"";
    my $tbeg = $start * $flen;
    my $dur = $end * $flen - $tbeg;

    $tbeg=sprintf "tbeg=\"%.${digits}f\"", $tbeg;
    $dur=sprintf  "dur=\"%.${digits}f\"", $dur;
    $score=sprintf "score=\"%.${digits}f\"", $score;
    my $channel="channel=\"1\"";

    print "    <kw $filename $channel $tbeg $dur $score $decision/>\n";
  }
  print "  </detected_kwlist>\n";
}

my $KWID="";
my @putative_hits;

print "<kwslist kwlist_filename=\"$kwlist_filename\" language=\"$language\" system_id=\"$system_id\">\n";

while (my $line = <STDIN>) {
  chomp $line;
  (my $kwid, my $file, my $start, my $end, my $score) = split " ", $line;

  if ($kwid ne $KWID) {
    PrettyPrint(\@putative_hits) if $KWID;
    $KWID=$kwid;
    @putative_hits = ();
  }

  push @putative_hits, [$kwid, $file, $start, $end, $score];

}
PrettyPrint(\@putative_hits) if $KWID;

print "</kwslist>\n"
