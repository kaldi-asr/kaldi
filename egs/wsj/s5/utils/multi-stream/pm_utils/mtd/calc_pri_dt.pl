#!/usr/bin/perl

my $tdlenlist = $ARGV[0];
my $transfile = $ARGV[1];

my @deltats = ();
open(IN, $tdlenlist) || die;
while (<IN>)
{
  chomp;
  @deltats = (@deltats,split(/[\s+,]/,$_));
}
close(IN);

##----------------------------------------------------------------------
##----------------------------------------------------------------------

my $nacc = 0;
my $nacc_delta = ();
for (my $i = 0; $i < @deltats; $i++){
  $nacc_delta[$i] = 0;
}
my $maxtd = $deltats[$#deltats];
# print "max:$maxtd\n";

open(IN, $transfile) || die;
while (<IN>)
{
  chomp;
  my @phones = split(' ', $_);
  # print "Processing $phones[0] ...\n";
  for (my $t = $maxtd+1; $t < @phones; ++$t)
  {
    for (my $i = 0; $i < @deltats; $i++){
      my $dt = $deltats[$i];
      if ($phones[$t] == $phones[$t-$dt]){
        $nacc_delta[$i]++;
      }
    }
    $nacc++;
  }
}
close(IN);

for (my $i = 0; $i < @deltats; $i++){
  $nacc_delta[$i] = $nacc_delta[$i] / $nacc;
}
for (my $i = 0; $i < @deltats; $i++){
  my $prob_same = $nacc_delta[$i];
  my $prob_diff = 1.0 - $prob_same;
  print "$prob_same $prob_diff\n";
}

##----------------------------------------------------------------------
##          End: calc_pri_dt.pl
##----------------------------------------------------------------------
