#!/usr/bin/perl -w
# compute_ug_costs.pl 
use strict;
use warnings;
use Carp;

my @A = ();
my $tot_count = 0;
my %count = ();
my $n_sent = 0;
my $p = 0.0;
my $cost = 0.0;
my $final_cost = 0.0;

while(my $line = <>) {
    chomp $line;
    my ($utt, $sent) = split /\t/, $line, 2;
    @A = split /\s+/, $sent;
    foreach my $w (@A) {
	$tot_count++;
	$count{$w}++;
    }
    $n_sent++;
}
$tot_count += $n_sent;
foreach my $k (keys %count) {
    $p = $count{$k} / $tot_count;
    $cost = -log($p);
    print "0  0  $k  $k  $cost\n";
}
$final_cost = -log($n_sent / $tot_count);
print "0 $final_cost\n";
