#!/usr/bin/env perl

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

#scalar(@ARGV) >= 1 && print STDERR "Usage: summarize_warnings.pl <log-dir>\n" && exit 1;

sub split_hundreds { # split list of filenames into groups of 100.
  my $names = shift @_;
  my @A = split(" ", $names);
  my @ans = ();
  while (@A > 0) {
    my $group = "";
    for ($x = 0; $x < 100 && @A>0; $x++) {
      $fname = pop @A;
      $group .= "$fname ";
    }
    push @ans, $group;
  }
  return @ans;
}

sub parse_accounting_entry {
  $entry= shift @_;

  @elems = split " ", $entry;
  
  $time=undef;
  $threads=undef;
  foreach $elem (@elems) {
    if ( $elem=~ m/time=(\d+)/ ) {
      $elem =~ s/time=(\d+)/$1/;
      $time = $elem;
    } elsif ( $elem=~ m/threads=(\d+)/ ) {
      $elem =~ s/threads=(\d+)/$1/g;
      $threads = $elem;
    } else {
      die "Unknown entry \"$elem\" when parsing \"$entry\" \n";
    }
  }

  if (defined($time) and defined($threads) ) {
    return ($time, $threads);
  } else {
    die "The accounting entry \"$entry\" did not contain all necessary attributes";
  }
}

foreach $dir (@ARGV) {

  #$dir = $ARGV[0];
  print $dir

  ! -d $dir && print STDERR "summarize_warnings.pl: no such directory $dir\n" ;

  $dir =~ s:/$::; # Remove trailing slash.


  # Group the files into categories where all have the same base-name.
  foreach $f (glob ("$dir/*.log")) {
    $f_category = $f;
    # do next expression twice; s///g doesn't work as they overlap.
    $f_category =~ s:\.\d+\.(?!\d+):.*.:;
    #$f_category =~ s:\.\d+\.:.*.:;
    $fmap{$f_category} .= " $f";
  }
}

foreach $c (sort (keys %fmap) ) {
  $n = 0;
  foreach $fgroup (split_hundreds($fmap{$c})) {
    $n += `grep -w WARNING $fgroup | wc -l`;
  }
  if ($n != 0) {
    print "$n warnings in $c\n"
  }
}
foreach $c (sort (keys %fmap)) {
  $n = 0;
  foreach $fgroup (split_hundreds($fmap{$c})) {
    $n += `grep -w ERROR $fgroup | wc -l`;
  }
  if ($n != 0) {
    print "$n errors in $c\n"
  }
}

$supertotal_cpu_time=0.0;
$supertotal_clock_time=0.0;
$supertotal_threads=0.0;

foreach $c (sort (keys %fmap)) {
  $n = 0;

  $total_cpu_time=0.0;
  $total_clock_time=0.0;
  $total_threads=0.0;
  foreach $fgroup (split_hundreds($fmap{$c})) {
    $lines=`grep -P "# Accounting:? " $fgroup |sed 's/.* Accounting:* *//g'`;
    
    #print $lines ."\n";

    @entries = split "\n", $lines;

    foreach $line (@entries) {
      $time, $threads = parse_accounting_entry($line);

      $total_cpu_time += $time * $threads;
      $total_threads += $threads;
      if ( $time > $total_clock_time ) {
        $total_clock_time = $time;
      }
    }
  }
  print "total_cpu_time=$total_cpu_time clock_time=$total_clock_time total_threads=$total_threads group=$c\n";

  $supertotal_cpu_time += $total_cpu_time;
  $supertotal_clock_time += $total_clock_time;
  $supertotal_threads += $total_threads;
}
print "total_cpu_time=$supertotal_cpu_time clock_time=$supertotal_clock_time total_threads=$supertotal_threads group=all\n";

