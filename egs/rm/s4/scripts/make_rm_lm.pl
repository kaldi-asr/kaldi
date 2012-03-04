#!/usr/bin/perl

# Copyright 2010-2011 Yanmin Qian  Microsoft Corporation

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

# This file takes as input the file wp_gram.txt that comes with the RM
# distribution, and creates the language model as an acceptor in FST form.

# make_rm_lm.pl   wp_gram.txt > G.txt

if (@ARGV != 1) {
    print "usage: make_rm_lm.pl  wp_gram.txt > G.txt\n";
    exit(0);
}
unless (open(IN_FILE, "@ARGV[0]")) {
    die ("can't open @ARGV[0]");
}


$flag = 0;
$count_wrd = 0;
$cnt_ends = 0;
$init = "";

while ($line = <IN_FILE>)
{	
	chop($line);

    $line =~ s/ //g;
    
	if(($line =~ /^>/)) 
	{
		if($flag == 0) 
		{
			$flag = 1;
		}
		$line =~ s/>//g;
		$hashcnt{$init} = $i;
		$init = $line;
		$i = 0;
		$count_wrd++;
		@LineArray[$count_wrd - 1] = $init;
 		$hashwrd{$init} = 0;
	}
	elsif($flag != 0)
	{
		
		$hash{$init}[$i] = $line;
		$i++; 			
		if($line =~ /SENTENCE-END/)
		{
			$cnt_ends++;
		}
 	} 
	else
	{}
}

$hashcnt{$init} = $i;

$num = 0;
$weight = 0;
$init_wrd = "SENTENCE-END";
$hashwrd{$init_wrd} = @LineArray;
for($i = 0; $i < $hashcnt{$init_wrd}; $i++)
{
	$weight = -log(1/$hashcnt{$init_wrd});
	$hashwrd{$hash{$init_wrd}[$i]} = $i + 1;
	print "0    $hashwrd{$hash{$init_wrd}[$i]}    $hash{$init_wrd}[$i]    $hash{$init_wrd}[$i]    $weight\n";
}
$num = $i;

for($i = 0; $i < @LineArray; $i++)
{
	if(@LineArray[$i] eq 'SENTENCE-END')
	{}
	else
	{
		if($hashwrd{@LineArray[$i]} == 0)
		{
			$num++;
			$hashwrd{@LineArray[$i]} = $num;
		}
		for($j = 0; $j < $hashcnt{@LineArray[$i]}; $j++)
		{
			$weight = -log(1/$hashcnt{@LineArray[$i]});
			if($hashwrd{$hash{@LineArray[$i]}[$j]} == 0)
			{
				$num++;
				$hashwrd{$hash{@LineArray[$i]}[$j]} = $num;
			}
			if($hash{@LineArray[$i]}[$j] eq 'SENTENCE-END')
			{
				print "$hashwrd{@LineArray[$i]}    $hashwrd{$hash{@LineArray[$i]}[$j]}    <eps>    <eps>    $weight\n"
                }
			else
			{
				print "$hashwrd{@LineArray[$i]}    $hashwrd{$hash{@LineArray[$i]}[$j]}    $hash{@LineArray[$i]}[$j]    $hash{@LineArray[$i]}[$j]    $weight\n";
			}
		}
	}
}

print "$hashwrd{$init_wrd}    0\n";
close(IN_FILE);


