#!/usr/bin/env perl
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

# This file takes as input the file pcdsril.txt that comes with the RM
# distribution, and creates the dictionary used in RM training.

# make_rm_dct.pl   pcdsril.txt > dct.txt

if (@ARGV != 1) {
    die "usage: make_rm_dct.pl   pcdsril.txt > dct.txt\n";
}
unless (open(IN_FILE, "@ARGV[0]")) {
    die ("can't open @ARGV[0]");
}

while ($line = <IN_FILE>)
{	
	chop($line);
	if (($line =~ /^[a-z]/)) 
	{
		$line =~ s/\+1//g;
		@LineArray = split(/\s+/,$line);
		@LineArray[0] = uc(@LineArray[0]);

		printf "%-16s",  @LineArray[0];
		for ($i = 1; $i < @LineArray; $i ++)
		{
			if (@LineArray[$i] eq 'q')
			{}
			elsif (@LineArray[$i] eq 'zh')
			{
				printf "sh ";
			}
			elsif (@LineArray[$i] eq 'eng')
			{
				printf "ng ";
			}
			elsif (@LineArray[$i] eq 'hv')
			{
				printf "hh ";
			}
			elsif (@LineArray[$i] eq 'em')
			{
				printf "m ";
			}
			elsif (@LineArray[$i] eq 'axr')
			{
				printf "er ";
			}
			elsif (@LineArray[$i] eq 'tcl')
			{
				if (@LineArray[$i+1] ne 't')
				{
					printf "td ";
				}
			}
			elsif (@LineArray[$i] eq 'dcl')
			{
				if (@LineArray[$i+1] ne 'd')
				{
					printf "dd ";
				}
			}
			elsif (@LineArray[$i] eq 'kcl')
			{
				if (@LineArray[$i+1] ne 'k')
				{
					printf "kd ";
				}
			}
			elsif (@LineArray[$i] eq 'pcl')
			{
				if (@LineArray[$i+1] ne 'p')
				{
					printf "pd ";
				}
			}
			elsif (@LineArray[$i] eq 'bcl')
			{
				if (@LineArray[$i+1] ne 'b')
				{
					printf "b ";
				}
			}
			elsif (@LineArray[$i] eq 'gcl')
			{
				if (@LineArray[$i+1] ne 'g')
				{
					printf "g ";
				}
			}
			elsif (@LineArray[$i] eq 't')
			{
				if (@LineArray[$i+1] ne 's')
				{
					printf "@LineArray[$i] ";
				}
				else
				{
					printf "ts ";
					$i++;
				}
			}
			else
			{
				printf "@LineArray[$i] ";
			}
		}
		printf "\n";
	}
}

printf "!SIL  sil\n";

close(IN_FILE);


