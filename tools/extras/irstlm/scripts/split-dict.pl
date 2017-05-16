#! /usr/bin/perl

#*****************************************************************************
# IrstLM: IRST Language Model Toolkit
# Copyright (C) 2007 Marcello Federico, ITC-irst Trento, Italy

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

#usage:
#split-dict.pl <input> <output> <parts>
#It splits the <input> dictionary into <parts> dictionaries
#(named <output000>, ... <output999>)
#splitting is balanced wrt to frequency of the <input> dictionary
#if not available a frequency of 1 is considered

use strict;
use Getopt::Long "GetOptions";
use File::Basename;

my ($help,$input,$output,$parts)=();

$help=1 unless
&GetOptions('input=s' => \$input,
            'output=s' => \$output, 
             'parts=i' => \$parts,           
             'h|help' => \$help,);

if ($help || !$input || !$output || !$parts) {
	my $cmnd = basename($0);
  print "\n$cmnd - splits a dictionary into frequency-balanced partitions\n",
	"\nUSAGE:\n",
	"       $cmnd [options]\n",
	"\nDESCRIPTION:\n",
	"       $cmnd splits a dictionary into frequency-balanced partitions.\n",
	"       The dictionary must be generated with IRSTLM command dict.\n",
	"       If dictionary does not contain frequencies, then a frequency 1 is\n",
	"       assumed for all words.\n",
	"\nOPTIONS:\n",
    "       --input <string>      input dictionary with frequencies\n",
    "       --output <string>     prefix of output dictionaries\n",
    "       --parts <int>         number of partitions to create\n",
    "       -h, --help            (optional) print these instructions\n",
    "\n";

  exit(1);
}



my $freqflag=0;
my ($w,$f,$globf,$thr);
my (@D,@F,%S,@C);
open(IN,"$input");

chomp($_=<IN>);
#if input is a dictionary.
if (/^dictionary[ \t]+\d+[ \t]+\d+$/i){
  my ($dummy,$size);
  ($dummy,$dummy,$size)=split(/[ \t]+/,$_);
  $freqflag=1 if /DICTIONARY/;
}

$globf=0;
while(chomp($_=<IN>)){
	if ($freqflag){
		($w,$f)=split(/[ \t]+/,$_);
	}
	else{
		$w=$_;
		$f=1;
	}
	push @D, $w;
	push @F, $f;
  $globf+=$f;
}
close (IN);

$thr=$globf/$parts;
my $totf=0;
print STDERR "Dictionary 0: (thr: $thr , $globf, $totf , $parts)\n";

my $sfx=0;
my $w;
for (my $i=0;$i<=$#D;$i++){
	
# if the remaining words are less than or equal to 
# the number of remaining sub-dictionaries to create
# put only one word per each sub-dictionary.
	if (($totf>0) && ($#D+1-$i) <= ($parts-1-$sfx)){
# recompute threshold on the remaining global frequency
# according to the number of remaining parts
		$sfx++;
		$globf-=$totf;
		$thr=($globf)/($parts-$sfx);
		print STDERR "Dictionary $sfx: (thr: $thr , $globf , $totf , ",($parts-$sfx),")\n";
		$totf=0;
	}

	$totf+=$F[$i];
	$w=$D[$i];
	$S{$w}=$sfx;
	$C[$sfx]++;
	if ($totf>$thr){
# recompute threshold on the remaining global frequency
# according to the number of remaining parts
		$sfx++;
		$globf-=$totf;
		$thr=($globf)/($parts-$sfx);
		print STDERR "Dictionary $sfx: (thr: $thr , $globf , $totf , ",($parts-$sfx),")\n";
		$totf=0;
	}
}


my $oldsfx=-1;
for (my $i=0;$i<=$#D;$i++){
	$w=$D[$i];
	$sfx="0000$S{$w}";
	$sfx=~s/.+(\d{3})/$1/;
	if ($sfx != $oldsfx){
#print STDERR "opening $output$sfx\n";
		close (OUT) if $oldsfx!= -1;
		open(OUT,">$output$sfx");
		if ($freqflag){
			print OUT "DICTIONARY 0 $C[$sfx]\n";
		}
		else{
			print OUT "dictionary 0 $C[$sfx]\n";
		}
		$oldsfx=$sfx;
	}
	if ($freqflag){
		print OUT "$w $F[$i]\n";
	}
	else{
		print OUT "$w\n";
	}
}
close (OUT) if $oldsfx!= -1;

my $numdict=$S{$D[$#D]}+1;
die "Only $numdict dictionaries were crested instead of $parts!" if ($numdict != $parts);

