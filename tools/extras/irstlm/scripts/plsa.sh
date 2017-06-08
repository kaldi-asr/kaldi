#! /bin/bash

set -m # Enable Job Control



function usage()
{
cmnd=$(basename $0);
cat<<EOF

$cmnd - train and/or test a probabilistic latent semantic model

USAGE:
$cmnd [options]

TRAINING OPTIONS:

-c file     Collection of training documents e.g. 'gunzip -c docs.gz'
-d file     Dictionary file (default dictionary)
-f          Force to use existing dictionary
-m fle      Output model file e.g. model
-n count    Number of topics (default 100)
-i count    Number of training iterations (default 20)
-t folder   Temporary working directory (default ./stat_PID)
-p count    Prune words with counts < arg (default 2)
-k count    Number of processes (default 5)

-r file     Model output file in readable format
-s count    Put top arg frequent words in special topic 0
-l file     Log file (optional)
-v          Verbose
-h          Show this message


TESTING OPTIONS

-c file     Testing documents e.g. test
-d file     Dictionary file (default dictionary)
-m file     Model file
-n number   Number of topics (default 100)
-u file     Output document unigram distribution
-o file     Output document topic distributions
-i counts   Number of training iterations (default 20)
-t folder   Temporary working directory (default ./stat_PID)
-l file     Log file (optional)
-k count    Number of processes (default 5)
-v          Verbose
-h          Show this message


EOF
}



if [ ! $IRSTLM ]; then
echo "Set IRSTLM environment variable with path to irstlm"
exit 2
fi

#paths to scripts and commands in irstlm
scr=$IRSTLM/bin
bin=$IRSTLM/bin
gzip=`which gzip 2> /dev/null`;
gunzip=`which gunzip 2> /dev/null`;

#default parameters
tmpdir=stat_$$
data=""
topics=100
splits=5
iter=20
prunefreq=2
spectopics=0
logfile="/dev/null"
verbose=""
unigram=""
outtopic=""
dict="dictionary"
forcedict=""
model=""
txtfile="/dev/null"

while getopts "hvfc:m:r:k:i:n:t:d:p:s:l:u:o:" OPTION
do
case $OPTION in
h)
usage
exit 0
;;
v)
verbose="--verbose";
;;
c)
data=$OPTARG
;;
m)
model=$OPTARG
;;
r)
txtfile=$OPTARG
;;
k)
splits=$OPTARG
;;
i)
iter=$OPTARG
;;
t)
tmpdir=$OPTARG
;;
d)
dict=$OPTARG
;;
f)
forcedict="TRUE"
;;
p)
prunefreq=$OPTARG
;;
s)
spectopics=$OPTARG
;;
n)
topics=$OPTARG
;;
l)
logfile=$OPTARG
;;
u)
unigram=$OPTARG
;;
o)
outtopic=$OPTARG
;;

?)
usage
exit 1
;;
esac
done

if [ $verbose ]; then
echo data=$data  model=$model  topics=$topics iter=$iter dict=$dict
logfile="/dev/stdout"
fi

if [ "$unigram" == "" -a "$outtopic" == "" ]; then

#training branch

if [ ! "$data" -o  ! "$model" ]; then
usage
exit 1
fi

if [ -e $logfile -a $logfile != "/dev/null" -a $logfile != "/dev/stdout" ]; then
echo "Logfile $logfile already exists! either remove or rename it."
exit 1
fi

if [ -e $model ]; then
echo "Output file $model already exists! either remove or rename it." >> $logfile 2>&1
exit 1
fi

if [ -e $txtfile -a $txtfile != "/dev/null" ]; then
echo "Output file $txtfile already exists! either remove or rename it." >> $logfile 2>&1
exit 1
fi


if [ -e $logfile -a $logfile != "/dev/null" -a $logfile != "/dev/stdout" ]; then
echo "Logfile $logfile already exists! either remove or rename it." >> $logfile 2>&1
exit 1
fi

#if [ ! -e "$data" ]; then
#echo "Cannot find data $data." >> $logfile 2>&1
#exit 1;
#fi

if [ ! -e $dict ]; then
echo extract dictionary >> $logfile
$bin/dict -i="$data" -o=$dict -PruneFreq=$prunefreq -f=y >> $logfile 2>&1
if [ `head -n 1 $dict| cut -d " " -f 3` -lt 10 ]; then
echo "Dictionary contains errors"
exit 2;
fi
else
echo "Warning: dictionary file already exists." >> $logfile 2>&1
if [ $forcedict ]; then
echo "Warning: authorization to use it." >> $logfile 2>&1
else
echo "No authorization to use it (see option -f)." >> $logfile 2>&1
exit 1
fi
fi



#check tmpdir
tmpdir_created=0;
if [ ! -d $tmpdir ]; then
echo "Creating temporary working directory $tmpdir" >> $logfile 2>&1
mkdir -p $tmpdir;
tmpdir_created=1;
else
echo "Cleaning temporary directory $tmpdir" >> $logfile 2>&1
rm $tmpdir/* 2> /dev/null
if [ $? != 0 ]; then
echo "Warning: some temporary files could not be removed" >> $logfile 2>&1
fi
fi

#####
echo split documents >> $logfile 2>&1
$bin/plsa -c="$data" -d=$dict -b=$tmpdir/data -sd=$splits >> $logfile 2>&1

machine=`uname -s` 
if [ $machine == "Darwin" ] ; then
splitlist=`jot - 1 $splits`
iterlist=`jot - 1 $iter`
else
splitlist=`seq  1 1 $splits`
iterlist=`seq 1 1 $iter`
fi

#rm $tmpdir/Tlist
for sp in $splitlist ; do echo $tmpdir/data.T.$sp >> $tmpdir/Tlist 2>&1; done
#rm $model
for it in $iterlist ; do
echo "ITERATION $it" >> $logfile 2>&1
for sp in $splitlist ; do
(date; echo it $it split $sp )>> $logfile 2>&1
$bin/plsa -c=$tmpdir/data.$sp -d=$dict -st=$spectopics -hf=$tmpdir/data.H.$sp -tf=$tmpdir/data.T.$sp -wf=$model -m=$model -t=$topics -it=1 -tit=$it >> $logfile 2>&1 &
done
while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done

(date; echo recombination ) >> $logfile 2>&1

$bin/plsa -ct=$tmpdir/Tlist -c="$data" -d=$dict -hf=$tmpdir/data.H -m=$model -t=$topics -it=1 -txt=$txtfile >> $logfile 2>&1

done
(date; echo End of training) >> $logfile 2>&1

echo "Cleaning temporary directory $tmpdir" >> $logfile 2>&1 
rm $tmpdir/* 2> /dev/null

if [ $tmpdir_created -eq 1 ]; then
echo "Removing temporary directory $tmpdir" >> $logfile 2>&1
rmdir $tmpdir 2> /dev/null
if [ $? != 0 ]; then
echo "Warning: the temporary directory could not be removed." >> $logfile 2>&1
fi
fi
exit 0

           
           
#testing branch
else

if [ ! $model -o ! -e $model ]; then
echo "Need to specify existing model" >> $logfile 2>&1
exit 1;
fi


if [ ! $dict  -o ! -e $dict  ]; then
echo "Need to specify dictionary file of the model" >> $logfile 2>&1
exit 1;
fi

if [ $unigram ]; then
$bin/plsa -inf="$data" -d=$dict -m=$model -hf=hfff.out$$ -t=$topics -it=$iter -wof=$unigram >> $logfile 2>&1
rm hfff.out$$

else  #topic distribution

#check tmpdir
tmpdir_created=0;
if [ ! -d $tmpdir ]; then
echo "Creating temporary working directory $tmpdir" >> $logfile 2>&1
mkdir -p $tmpdir;
tmpdir_created=1;
else
echo "Cleaning temporary directory $tmpdir" >> $logfile 2>&1
rm $tmpdir/* 2> /dev/null
if [ $? != 0 ]; then
echo "Warning: some temporary files could not be removed" >> $logfile 2>&1
fi
fi

#####
echo split documents >> $logfile 2>&1
$bin/plsa -c="$data" -d=$dict -b=$tmpdir/data -sd=$splits >> $logfile 2>&1

machine=`uname -s`
if [ $machine == "Darwin" ] ; then
splitlist=`jot - 1 $splits`
else
splitlist=`seq 1 1 $splits`
fi

#rm $tmpdir/Tlist
for sp in $splitlist ; do echo $tmpdir/data.T.$sp >> $tmpdir/Tlist 2>&1; done
#rm $model

for sp in $splitlist ; do
(date; echo split $sp )>> $logfile 2>&1

$bin/plsa -inf=$tmpdir/data.$sp -d=$dict -hf=$tmpdir/data.H.$sp -m=$model -t=$topics -it=$iter -tof=$tmpdir/topic.$sp >> $logfile 2>&1 &

done
while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done

(date; echo recombination ) >> $logfile 2>&1

echo > $outtopic
for sp in $splitlist ; do  #makes sure that 1 < 2 < ... < 11 ...
cat $tmpdir/topic.$sp >> $outtopic
done

(date; echo End of training) >> $logfile 2>&1

echo "Cleaning temporary directory $tmpdir" >> $logfile 2>&1
rm $tmpdir/* 2> /dev/null

if [ $tmpdir_created -eq 1 ]; then
echo "Removing temporary directory $tmpdir" >> $logfile 2>&1
rmdir $tmpdir 2> /dev/null
if [ $? != 0 ]; then
echo "Warning: the temporary directory could not be removed." >> $logfile 2>&1
fi
fi

fi
fi


exit 0


