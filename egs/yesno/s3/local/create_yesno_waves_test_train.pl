#!/usr/bin/perl

$full_list = $ARGV[0];
$test_list = $ARGV[1];
$train_list = $ARGV[2];

open FL, $full_list;
$nol = 0;
while ($l = <FL>)
{
	$nol++;
}
close FL;

$i = 0;
open FL, $full_list;
open TESTLIST, ">$test_list";
open TRAINLIST, ">$train_list";
while ($l = <FL>)
{
	chomp($l);
	$i++;
	if ($i <= $nol/2 )
	{
		print TRAINLIST "$l\n";
	}
	else
	{
		print TESTLIST "$l\n";
	}
}
