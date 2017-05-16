#! /usr/bin/perl

#*****************************************************************************
# IrstLM: IRST Language Model Toolkit
# Copyright (C) 2010 Marcello Federico, FBK-irst Trento, Italy

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

#******************************************************************************
#Sorts n-grams of an ARPA file according to lexicographic order.
#Inverted sorting option is propedeutic to building a binary
#lmtable with compile-lm with n-grams stored in reverted order.

use strict;
use Getopt::Long "GetOptions";

my ($help,$ilm,$olm,$inv,$tmpdir)=();
$help=0;

$ilm="/dev/stdin";
$olm="/dev/stdout";
my $tmpdir="$ENV{TMP}";

&GetOptions('ilm=s' => \$ilm,
			'olm=s' => \$olm,
			'tmpdir=s' => \$tmpdir,
            'inv' => \$inv,
            'help' => \$help,);

if ($help || !$ilm || !$olm){
  print "sort-lm.pl [--ilm <fname>]  [--olm <fname>] [--inv]\n",
  "-ilm  <fname>   input ARPA LM filename (default /dev/stdin)\n",
  "-olm <fname>    output ARPA LM filename (default /dev/stdout)\n",
  "-tmpdir         temporary directory for sorting (default is the enivronment variable TMP\n",
  "-inv            inverted n-gram sort for compile-lm\n",
  "-help           print these instructions\n";    
  exit(1);
}

warn "temporary directory for sorting is $tmpdir\n";

my $order=0;
my $sortcmd="";

$ENV{'LC_ALL'}='C';

open (INP, "< $ilm") || die "cannot open input LM file: $ilm\n";
open (OUT, "> $olm") || die "cannot open output LM file: $olm\n";


warn "reading from standard input\n" if $ilm eq "/dev/stdin";
warn "writing to standard output\n" if $olm eq "/dev/stdout";

$_=<INP>;

#sanity check
die "Error: input cannot be an intermediate iARPA file. First convert it to ARPA format with compile-lm.\n" if 
$_=~/^iARPA/;

my $isQuantized=0;
$isQuantized=1 if $_=~/^qARPA/;

while(!/^\\end\\/){

	
	if (($order)=$_=~/^\\(\d+)-grams:/){
		print(OUT $_);$_=<INP>;	
		if ($isQuantized){
			print(OUT $_); chop $_;#print centers
			my $centers=$_; $_=<INP>;
			warn "skip $centers centers\n";		
			for (my $c=1;$c<=$centers;$c++){
				print(OUT $_);$_=<INP>; 
			}
			
		}
		#sort command
		#$sortcmd="sort -b"; #does not seem to work properly
		$sortcmd="sort --temporary-directory=$tmpdir";
		if ($inv){
			warn "inverted sorting of $order-grams\n";
			for (my $n=$order;$n>0;$n--){
				$sortcmd.=" -k ".($n+1).",".($n+1);
			}
		}else{
			warn "direct sorting of $order-grams\n";
			for (my $n=1;$n<=$order;$n++){
				$sortcmd.=" -k ".($n+1).",".($n+1);
			}
		}
				
		close(OUT);open (OUT,"|$sortcmd >> $olm");
		
		
		do{ 
			print(OUT $_);$_=<INP>;			
				
		}until (/^\\/ || /^\n/);
		
		close(OUT); open(OUT, ">> $olm");	
		
	}
	else{
		print(OUT $_);$_=<INP>;	
	}
	
}

print(OUT $_);

close(INP);
close(OUT);
