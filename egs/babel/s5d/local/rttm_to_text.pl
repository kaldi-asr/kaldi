#!/usr/bin/env perl

use warnings;
use strict;
use utf8;

use Data::Dumper;

sub float_gt {
  my ($A, $B) = @_;
  #print Dumper(\@_);

  if ( ($A - $B) < 1e-12 ) {
    return 0;
  } elsif ($A > $B ) {
    return 1;
  } else {
    return 0;
  }
}

binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

my $datadir=$ARGV[0];
my $rttm_filename=$ARGV[1];


my $filename="";
my %rttm;
my @times;


open(rttm_f, "<:utf8", $rttm_filename) or die "Cannot open the RTTM file";
while ( <rttm_f> ) {
  chop;
  my @elems = split;
  my $_filename= $elems[1];
  my $_time=$elems[3];
  my $_dur=$elems[4];
  my $_text=$elems[5];

  #We could simply pull-out the vector of times
  #from the hash, but in case the RTTM is not sorted
  #there might be some other problem somewhere
  #(as the RTTMs are normally sorted). So instead of being
  #"smart", let's make the user notice!
  if ( exists($rttm{$_filename}) ) {
    die "The RTTM file is not sorted!";
  }

  if ( $filename ne $_filename ) {
    if ( $filename ne "" ) {
      #print $filename . "\n";
      my @tmp = @times;
      $rttm{$filename} = \@tmp;
      #if ($filename eq "BABEL_BP_101_10470_20111118_172644_inLine" ) {
      #  print "$filename\n";
      #  print Dumper($rttm{$filename});
      #}
      #print Dumper($rttm{"BABEL_BP_101_10470_20111118_172644_inLine"});
    }

    @times = ();
    $filename = $_filename;
  }

  #I don't really know what is the distinction between all
  #of these. Let's throw away the SPEAKER, as it does not
  #really contain information that is to be found in the transcript
  #and keep the others
  if ( $elems[0] eq "LEXEME") {
    push @times, [$_time, $_time + $_dur, $_text];
  } elsif ( $elems[0] eq "NON-SPEECH" ) {
    push @times, [$_time, $_time + $_dur, $_text];
  } elsif ( $elems[0] eq "NON-LEX" ) {
    push @times, [$_time, $_time + $_dur, $_text];
  } elsif ( $elems[0] eq "SPEAKER") {
    ;
  } else {
    #This is just a safety precaution if a new flag/type appears.
    die  "Unknown first element $elems[0] of line '" . join(" ", @elems) . "'\n";
  }

  #We compare the two last entries of the #times vector, if they
  #are ordered properly. Again, this is just a safety recaution
  #In a well-formed RTTM, this is normal.
  if ( (@times > 1) &&  float_gt($times[-2][1],  $times[-1][0])  ) {
    #print Dumper(\@times);
    my $A = $times[-2][0];
    my $B = $times[-1][0];
    my $Aend = $times[-2][1];
    my $Bend = $times[-1][1];

    #print  "WARNING: Elements in the RTTM file are not sorted for FILENAME $filename!\n";
    #print $times[-2][0] . " " . $times[-2][1] - $times[-2][0]. " " . $times[-2][2] . "\n";
    #print $times[-1][0] . " " . $times[-1][1] - $times[-1][0]. " " . $times[-1][2] . "\n";
    #print "\n";

    my @sorted =  sort {$a <=> $b} ($A, $B, $Aend, $Bend);
    #print Dumper(\@sorted);
    $times[-1][0] = $sorted[0];
    $times[-1][1] = $sorted[2]; #We omit the gap between these two words
    $times[-2][0] = $sorted[2];
    $times[-2][1] = $sorted[3];

  }
}
if ( $filename ne "" ) {
  #print $filename . "\n";
  $rttm{$filename} = \@times;
}
close(rttm_f);

open(segments_f, "<:utf8", "$datadir/segments") or die "Cannot open file $datadir/segments";
while ( <segments_f> ) {
  chop;
  my ($segmentname, $filename, $start, $end) = split;

  if (! exists $rttm{$filename} ) {
    print "Filename $filename does not exists in the RTTM file\n";
    die;
  }
  my @times = @{$rttm{$filename}};
  my $i;
  my $j;


  #if ($segmentname ne "10470_A_20111118_172644_000000" ) {
  #  next;
  #}

  #print $filename . "\n";

  #print Dumper(\@times);
  $i = 0;
  #print $start . " " . $times[$i][0] . " " . $times[$i][1] . "\n";
  while (($i < @times) && ( $times[$i][1] < $start ) ) { $i += 1; };
  $j = $i;
  while (($j < @times) && ( $times[$j][0] < $end ) ) { $j += 1; };

  print $segmentname . " ";
  while ( $i < $j ) {
    #print Dumper($times[$i]);
    print $times[$i][2] . " ";
    $i += 1;
  }
  print "\n";
  #die
}
close(segments_f);
