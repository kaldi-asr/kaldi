#! /bin/bash 

#/******************************************************************************
#IrstLM: IRST Language Model Toolkit
#Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy
#
#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foundation; either
#version 2.1 of the License, or (at your option) any later version.
#
#This library is distributed in the hope that it will be useful,
# 
# 
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Lesser General Public License for more details.
#
#You should have received a copy of the GNU Lesser General Public
#License along with this library; if not, write to the Free Software
#Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
#
#******************************************************************************/

# mdtsel.sh
# by M. Federico
# Copyright Marcello Federico, Fondazione Bruno Kessler, 2012


set -m #enable job control

usage()
{
    cmnd=$(basename $0);
    cat << EOF

$cmnd - performs data selection assuming an indomain corpus and
        a very large out of domain corpus.

USAGE:
       $cmnd [options]

DESCRIPTION.
       This command performs data selection assuming an indomain
       corpus and a very large out of domain corpus.
       Both corpora must contain one sentence in each line delimited
       with <s> and </s>. The process produces a file of scores.


OPTIONS:
       -h        Show this message
       -v        Verbose
       -i        In-domain corpus 
       -o        Out-domain corpus
       -s        Scores output file 
       -x        Out-domain lines are indexed
       -w        Temporary work directory (default /tmp)
       -j        Number of jobs (default 6)
       -m        Data selection model (1 or 2, default 2)
       -f        Word frequency threshold (default 2)
       -n        Ngram order to use (n>=1 default 3)
       -d        Vocabulary size upper bound (default 10000000)   
       -c        Cross-validation parameter (cv>=1, default 1)

EOF
}


if [ ! $IRSTLM ]; then
   echo "Set IRSTLM environment variable with path to irstlm"
   exit 2
fi

#paths to scripts and commands in irstlm
scr=$IRSTLM/bin
bin=$IRSTLM/bin

#check irstlm installation
if [ ! -e $bin/dtsel ]; then
   echo "$IRSTLM does not contain a proper installation of IRSTLM"
   exit 3
fi

#default parameters
indomfile="";
outdomfile="";
scoresfile="";
workdir=/tmp
logfile="/dev/null"
jobs=6
model=2
minfreq=2
ngramorder=3
cv=1
dub=10000000

verbose="";
useindex=0;

while getopts “hvi:o:s:l:w:j:m:f:n:c:d:x:” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         v)
             verbose="--verbose";
             ;;
         i)
             indfile=$OPTARG
             ;;
			 
         o)
             outdfile=$OPTARG
             ;;
         s)
             scorefile=$OPTARG
             ;;			 
         l)
             logfile=$OPTARG
             ;;
         w)
		     workdir=$OPTARG
             ;;			 
         j)
		     jobs=$OPTARG
             ;;

		 m)
             model=$OPTARG
             ;;	 

         n)
             ngramorder=$OPTARG
             ;;
         f)
		     minfreq=$OPTARG;	
			 ;;
	     d)
		     dub=$OPTARG;	
			 ;;
		 x)
		     useindex=$OPTARG;	
			 ;;

 		 ?)
             usage
             exit 1
             ;;
	
		esac
done


if [ $verbose ];then
echo indfile= $indfile outdfile= $outdfile scorefile= $scorefile useindex= $useindex 
echo logfile= $logfile workdir= $workdir 
echo jobs= $jobs model= $model ngramorder= $ngramorder minfreq= $minfreq dub=$dub
fi

if [ ! $indfile -o ! $outdfile -o ! $scorefile ]; then
    usage
    exit 5
fi
 
if [ -e $scorefile ]; then
   echo "Output score file $outfile already exists! either remove or rename it."
   exit 6
fi

if [ $logfile != "/dev/null" -a $logfile != "/dev/stdout" -a -e $logfile ]; then
   echo "Logfile $logfile already exists! either remove or rename it."
   exit 7
fi

workdir_created=0

if [ ! -d $workdir ]; then
   echo "Temporary work directory $workdir does not exist";
   echo "creating $workdir";
   mkdir -p $workdir;
   workdir_created=1;
fi


#get process id to name process specific temporary files
pid=$$

#compute size of out domain corpus and block size of split
lines=`wc -l < $outdfile`
size=`echo "( $lines + 1000 )" / $jobs | bc` #to avoid any small block

#perform split 
split -l $size $outdfile $workdir/dtsel${pid}-files-

for file in $workdir/dtsel${pid}-files-*
do
echo $file  
( \
$bin/dtsel -x=$useindex -i=$indfile -o=$file -s=${file}.scores -n=$ngramorder -dub=$dub -f=$minfreq -m=$model ; \
cat ${file}.scores | perl -pe '/^nan /1000 /g;' | sort -g > ${file}.scores.tmp ; \
mv ${file}.scores.tmp ${file}.scores \ 
) >>$logfile 2>&1 &

done

# Wait for all parallel jobs to finish
while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done

sort -g -m $workdir/dtsel${pid}-files-*.scores > $scorefile
rm $workdir/dtsel${pid}-files-*
if [ $workdir_created == 1 ]
then
rmdir $workdir
fi



