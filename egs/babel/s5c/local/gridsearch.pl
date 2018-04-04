#!/usr/bin/env perl

use warnings;
use strict;
use List::Util qw(reduce);
use Data::Dumper;

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

sub escape {
  my @cmd_in = @{$_[0]};
  my @cmd = ();
  foreach my $x (@cmd_in) {
    if ($x =~ m/^\S+$/) { push @cmd, $x } # If string contains no spaces, take
                                          # as-is.

    elsif ($x =~ m:\":) { push @cmd,  "'\''$x'\'' "; } # else if no dbl-quotes, use single
    else { push @cmd,  "\"$x\" "; }  # else use double.
  }
  return @cmd;
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

  } elsif ($ARGV[$i] eq "-train") {
    if ( $cmdid ) {
      if ( $cmdid eq "-eval" ) {
        @evalcmd = @cmd;
      } elsif ( $cmdid eq "-train" ) {
        @traincmd = @cmd;
      }
    }

    $cmdid = $ARGV[$i];
    @cmd = ();

  } elsif ($ARGV[$i] eq "-eval") {
    if ( $cmdid ) {
      if ( $cmdid eq "-eval" ) {
        @evalcmd = @cmd;
      } elsif ( $cmdid eq "-train" ) {
        @traincmd = @cmd;
      }
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

if ( $cmdid ) {
  if ( $cmdid eq "-eval" ) {
    @evalcmd = @cmd;
  } elsif ( $cmdid eq "-train" ) {
    @traincmd = @cmd;
  }
}


my @combs;
@combs = cartesian_product( values %VARIABLES );
@combs =@{$combs[0]};
#print Dumper(@{$combs[0]});


#@combs = gen_combinations(values %VARIABLES);
#print Dumper(\@combs);
#@traincmd = escape(\@traincmd);
#@evalcmd = escape(\@evalcmd);


foreach my $comb (@combs) {
  my %params;
  @params{keys %VARIABLES} = @{$comb};

  my @out;
  @out = substitute(\@traincmd, \%params);
  print "Running train:\n" . join(" ", @out) . "\n";
  system(@out) == 0 or die "system @out failed: exit code $?";


  @out = substitute(\@evalcmd, \%params);
  print "Running eval:\n" . join(" ", @out) . "\n";
  system(@out) == 0 or die "system @out failed: exit code $?";

}



