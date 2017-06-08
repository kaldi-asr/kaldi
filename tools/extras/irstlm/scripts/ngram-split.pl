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

#re-segment google n-gram count files into files so that
#n-grams starting with a given word (prefix) are all 
#contained in one file.

use Getopt::Long "GetOptions";
use File::Basename;

my ($help,$lm,$size,$sublm)=();
$help=1 unless
&GetOptions('h|help' => \$help);

if ($help) {
	my $cmnd = basename($0);
  print "\n$cmnd - re-segment google n-gram count files so that n-grams\n",
    "       starting with a given word (prefix) are all contained in one file\n",
	"\nUSAGE:\n",
	"       $cmnd [options] [<output_prefix>]\n",
	"\nDESCRIPTION:\n",
	"       Input is expected on STDIN.\n",
	"       <output_prefix>       prefix of files to be created\n",
	"\nOPTIONS:\n",
    "       -h, --help            (optional) print these instructions\n",
    "\n";

  exit(1);
}


$max_pref=10000;   #number of prefixes to be put in one file 
$max_ngram=5000000;#number of n-grams to be put in one file
$file_cnt=0;       #counter of files 
$pref_cnt=0;       #counter of prefixes in the current file
$ngram_cnt=0;      #counter of n-gram in the current file   

$path=($ARGV[0]?$ARGV[0]:"goong");     #path of files to be created

$gzip=`which gzip`; 
chomp($gzip);

$pwrd="";
open(OUT,sprintf("|$gzip -c > %s.%04d.gz",$path,++$file_cnt));

while ($ng=<STDIN>){
  ($wrd)=$ng=~/^([^ ]+)/;
  #warn "$wrd\n";
  if ($pwrd ne $wrd){
    $pwrd=$wrd;
    if ($file_pref>$max_pref || $ngram_cnt>$max_ngram){
      warn "it's time to change file\n";
      close(OUT);
      open(OUT,sprintf("|$gzip -c > %s.%04d.gz",$path,++$file_cnt));
      $pref_cnt=$ngram_cnt=0;
    }
    else{
      $pref_cnt++;
    }
  }
  print OUT $ng;
  $ngram_cnt++;
}
close(OUT);

