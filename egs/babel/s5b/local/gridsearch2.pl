#! /usr/bin/perl

use warnings;
use strict;


use Data::Dump qw(pp dumpf);
use List::Util qw(reduce);

sub cartesian_product {
  reduce {
    [ map {
      my $item = $_;
      map [ @$_, $item ], @$a
    } @$b ]
  } [[]], @_
}


sub gen_range {
  (my $begin,  my $step, my $end) = split(':', $_[0]);

  my @range = ();
  for (my $i=$begin; $i <=$end; $i += $step) {
    push @range, $i;
  }

  return @range;
}

sub gen_sequence {
  my $name=$_[0];

  my @steps = split(',', $_[1]);
  my @seq=();

  foreach my $step (@steps) {
    if ($step =~/\d+:\d+:\d/) {
      push @seq, gen_range($step);
    } elsif ($step =~ /\d+/ ) {
      push @seq, $step;
    } else {
      die "Unsupported range atom $step in range spec $name";
    }
  }

  return ($name, @seq);
}

sub gen_combinations {

  my @combinations = ();

  foreach my $i  ( @{$_[0]} )  {
    foreach my $j ( @{$_[1]} ) {
      push @combinations, [$i, $j];
    }
  }
  return @combinations;
}

sub substitute {
  my @cmd_proto = @{$_[0]};
  my %valhash = %{$_[1]};


  my @cmd_out;

  foreach my $elem (@cmd_proto) {
    foreach my $key (keys %valhash) {
      #print $elem . "($key, " . $valhash{$key}. ")->";
      $elem =~ s/$key/$valhash{$key}/g;
      #print $elem . "\n";
    }
    push @cmd_out, $elem;
  }

  return @cmd_out
}


my %VARIABLES=();
my @cmd = ();
my $cmdid = undef;
my @traincmd = ();
my @evalcmd = ();
my @scorecmd = ();

my @known_switches = ("-train", "-eval", "-score");
my %found_switches = ();

for (my $i=0; $i < scalar(@ARGV); $i++) {
  if ($ARGV[$i] eq "-var") {
     
    $i++;
    (my $name, my @range) = gen_sequence(split('=', $ARGV[$i]));
    $VARIABLES{$name}=\@range
  
  } elsif (grep {$_ eq $ARGV[$i]} @known_switches) {

    if ($cmdid) {
      print "CMD: $cmdid\n";
      my @tmp = @cmd;
      $found_switches{$cmdid} = \@tmp;      
      pp(%found_switches);
    }

    $cmdid = "$ARGV[$i]";
    @cmd = ();

  } else {
    if ( $cmdid ) {
      push @cmd, $ARGV[$i];
    } else {
      die "Unknown option or switch '$ARGV[$i]' \n";
    }
  }
}

if ($cmdid) {
  print "CMD: $cmdid\n";
  my @tmp = @cmd;
  $found_switches{$cmdid} = \@tmp;      
}

pp(%VARIABLES);
pp(%found_switches);

my @combs = gen_combinations(values %VARIABLES);


foreach my $comb (@combs) {
  my %params;
  @params{keys %VARIABLES} = @{$comb};

  my @out;
  @out = substitute(\@traincmd, \%params);
  system(@out) == 0 or die "system @out failed: exit code $?";
  

  @out = substitute(\@evalcmd, \%params);
  system(@out) == 0 or die "system @out failed: exit code $?";
  
}



