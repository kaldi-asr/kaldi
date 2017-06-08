#! /bin/bash

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

function usage()
{
    cmnd=$(basename $0);
    cat<<EOF

$cmnd - adds sentence start/end symbols in each line and trims very very long words

USAGE:
       $cmnd [options]

OPTIONS:
       -h        Show this message
       -r count  Specify symbol repetitions (default 1)
       -t length Trim words up to _length_ chars (default 80)
       -s char   Specify symbol (default s)

EOF
}

#default setting
repeat=1; 
maxwordlen=80;
symbol="s"

# Parse options
while getopts "hr:t:s:" OPT; do
    case "$OPT" in
        h)
            usage >&2;
            exit 0;
            ;;
        r)  repeat=$OPTARG
            ;; 
        t)  maxwordlen=$OPTARG
            ;; 
        s)  symbol=$OPTARG
            ;; 
    esac
done

#adds start/end symbols to standard input and 
#trims words longer than 80 characters
eos="";
bos="";

for i in `seq $repeat`; do bos="$bos<${symbol}> "; eos="$eos <\/${symbol}>";done

(sed "s/^/$bos/" | sed "s/\$/ $eos/";) |\
sed "s/\([^ ]\{$maxwordlen\}\)\([^ ]\{1,\}\)/\1/g"

