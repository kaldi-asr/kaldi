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
#computes LM statistics over a string

use strict;
use Getopt::Long "GetOptions";
use File::Basename;

my ($help,$lm,$txt)=();
$help=1 unless

&GetOptions('lm=s' => \$lm,
            'txt=s' => \$txt,
            'h|help' => \$help,);

if ($help || !$lm || !$txt) {
	my $cmnd = basename($0);
  print "\n$cmnd - computes LM statistics over a string\n",
	"\nUSAGE:\n",
	"       $cmnd [options]\n",
	"\nOPTIONS:\n",
    "       --lm  <string>        language model file \n",
    "       --txt <string>        text file\n",
    "       -h, --help            (optional) print these instructions\n",
    "\n";

  exit(1);
}

if (!$ENV{IRSTLM}){
  print "Set environment variable IRSTLM with path to the irstlm directory\n";
  exit(1);
}



my $clm="$ENV{IRSTLM}/bin/compile-lm";

open (OUT,"$clm $lm --eval $txt --debug 1|");
while (<OUT>){
print;
}

close(OUT);
