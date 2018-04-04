#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;
use XML::Simple;

my $data = XMLin(\*STDIN);
my $duptime= $ARGV[0];

#print Dumper($data);

# Filters duplicate keywords that have the same keyword and about the same time.
# Relies on the fact that its input is sorted from largest to smallest score.

foreach my $kwentry (@{$data->{detected_kwlist}}) {
  #print "$kwentry->{kwid}\n";
  my $prev_time;
  my $prev_file;

  if(ref($kwentry->{kw}) eq 'ARRAY'){
    my @arr = @{$kwentry->{kw}};
    my @newarray = ();
    
    push @newarray, $arr[0];
    #$arr[0]->{tbeg} . "\n";
    for (my $i = 1; $i < scalar(@arr); $i +=1) {
      
      my $found = 0;
      foreach my $kw (@newarray) {
        if (( abs($arr[$i]->{tbeg} -  $kw->{tbeg}) < $duptime )  && 
            ( $arr[$i]->{channel} ==  $kw->{channel}) &&
            ( $arr[$i]->{file} eq  $kw->{file}) ) {

          $found = 1;
          
        #print $arr[$i]->{tbeg} . "\n";
        }
      }
      if ( $found == 0 ) {
        push @newarray, $arr[$i];
      }
    }

    $kwentry->{kw} = \@newarray;
  }else{
      #print $kwentry->{kw}->{tbeg} . "\n";
  }
#  print "$kwentry->{kwid}\t$kwentry->{kwtext}\n";
}
my $xml = XMLout($data, RootName => "kwslist", NoSort=>1);
print $xml;
