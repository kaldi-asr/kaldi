#! /bin/bash

function usage()
{
    cmnd=$(basename $0);
    cat<<EOF

$cmnd - estimates a language model file

USAGE:
       $cmnd [options]

OPTIONS:
       -h        Show this message
       -i        Input training file e.g. 'gunzip -c train.gz'
       -o        Output gzipped LM, e.g. lm.gz
       -k        Number of splits (default 5)
       -n        Order of language model (default 3)
       -t        Directory for temporary files (default ./stat_PID)
       -p        Prune singleton n-grams (default false)
       -u        Use uniform word frequency for dictionary splitting (default false)
       -q        Parameters for qsub ("-q <queue>", and any other)
       -s        Smoothing methods: witten-bell (default), kneser-ney (approximated kneser-ney), improved-kneser-ney
       -b        Include sentence boundary n-grams (optional)
       -d        Define subdictionary for n-grams (optional)
       -v        Verbose

EOF
}

hostname=`uname -n`
if [ $hostname == "voxgate" ] ; then
echo "voxgate can not be used as submission host"
echo "use any other cluster machine"
exit
fi

if [ ! $IRSTLM ]; then
   echo "Set IRSTLM environment variable with path to irstlm"
   exit 2;
fi

#paths to scripts and commands in irstlm
scr=$IRSTLM/bin
bin=$IRSTLM/bin
gzip=`which gzip 2> /dev/null`;
gunzip=`which gunzip 2> /dev/null`;

#check irstlm installation
if [ ! -e $bin/dict -o  ! -e $scr/split-dict.pl ]; then
   echo "$IRSTLM does not contain a proper installation of IRSTLM"
   exit 3;
fi

#default parameters
logfile=/dev/null
tmpdir=stat_$$
order=3
parts=3
inpfile="";
outfile=""
verbose="";
smoothing="--witten-bell";
prune="";
boundaries="";
dictionary="";
uniform="-f=y";
queueparameters=""

while getopts "hvi:o:n:k:t:s:q:pbl:d:u" OPTION
do
     case $OPTION in
         h)
             usage
             exit 0
             ;;
         v)
             verbose="--verbose";
             ;;
         i)
             inpfile=$OPTARG
             ;;
         d)
             dictionary="-sd=$OPTARG"
             ;;

         u)
             uniform=" "
             ;;

         o)
             outfile=$OPTARG
             ;;
         n)
             order=$OPTARG
             ;;
         k)
             parts=$OPTARG
             ;;
         t)
             tmpdir=$OPTARG
             ;;
         s)
             smoothing=$OPTARG
	     case $smoothing in
	     witten-bell) 
		     smoothing="--witten-bell"
		     ;; 
	     kneser-ney)
		     smoothing="--kneser-ney"
		     ;;
             improved-kneser-ney)
                     smoothing="--improved-kneser-ney"
                     ;;
	     *) 
		 echo "wrong smoothing setting";
		 exit 4;
	     esac
             ;;
         p)
             prune='--prune-singletons';
             ;;
         q)
             queueparameters=$OPTARG;
             ;;
         b)
             boundaries='--cross-sentence';
             ;;
	 l)
             logfile=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done


if [ $verbose ]; then
echo inpfile=\"$inpfile\" outfile=$outfile order=$order parts=$parts tmpdir=$tmpdir prune=$prune smoothing=$smoothing dictionary=$dictionary verbose=$verbose
fi

if [ ! "$inpfile" -o ! "$outfile" ]; then
    usage
    exit 5 
fi
 
if [ -e $outfile ]; then
   echo "Output file $outfile already exists! either remove or rename it."
   exit 6;
fi

if [ -e $logfile -a $logfile != "/dev/null" -a $logfile != "/dev/stdout" ]; then
   echo "Logfile $logfile already exists! either remove or rename it."
   exit 7;
fi

#check tmpdir
tmpdir_created=0;
if [ ! -d $tmpdir ]; then
   echo "Temporary directory $tmpdir does not exist";
   echo "creating $tmpdir";
   mkdir -p $tmpdir;
   tmpdir_created=1;
else
   echo "Cleaning temporary directory $tmpdir";
   rm $tmpdir 2> /dev/null
   if [ $? != 0 ]; then
      echo "Warning: some temporary files could not be removed"
   fi
fi

workingdir=`pwd | perl -pe 's/\/nfsmnt//g'`
cd $workingdir

qsubout="$workingdir/DICT-OUT$$"
qsuberr="$workingdir/DICT-ERR$$"
qsublog="$workingdir/DICT-LOG$$"
qsubname="DICT"

(\
qsub $queueparameters -b no -sync yes -o $qsubout -e $qsuberr -N $qsubname << EOF
cd $workingdir
echo exit status $?
echo "Extracting dictionary from training corpus"
$bin/dict -i="$inpfile" -o=$tmpdir/dictionary $uniform -sort=no
echo exit status $?
echo "Splitting dictionary into $parts lists"
$scr/split-dict.pl --input $tmpdir/dictionary --output $tmpdir/dict. --parts $parts
echo exit status $?
EOF
) 2>&1 > $qsublog

unset suffix
#getting list of suffixes
for file in `ls $tmpdir/dict.*` ; do
sfx=`echo $file | perl -pe 's/^.+\.(\d+)$/$1/'`
suffix[${#suffix[@]}]=$sfx
done

qsubout="$workingdir/NGT-OUT$$"
qsuberr="$workingdir/NGT-ERR$$"
qsublog="$workingdir/NGT-LOG$$"
qsubname="NGT"

unset getpids
echo "Extracting n-gram statistics for each word list"
echo "Important: dictionary must be ordered according to order of appearance of words in data"
echo "used to generate n-gram blocks,  so that sub language model blocks results ordered too"

for sfx in ${suffix[@]} ; do

(\
qsub $queueparameters -b no -j yes -sync no -o $qsubout.$sfx -e $qsuberr.$sfx -N $qsubname-$sfx << EOF
cd $workingdir
echo exit status $?
$bin/ngt -i="$inpfile" -n=$order -gooout=y -o="$gzip -c > $tmpdir/ngram.dict.${sfx}.gz" -fd="$tmpdir/dict.${sfx}" $dictionary -iknstat="$tmpdir/ikn.stat.dict.${sfx}" 
echo exit status $?
echo
EOF
) 2>&1 > $qsublog.$sfx

id=`cat $qsublog.$sfx | grep 'Your job' | awk '{print $3}'`
sgepid[${#sgepid[@]}]=$id

done

waiting=""
for id in ${sgepid[@]} ; do waiting="$waiting -hold_jid $id" ; done

qsub $queueparameters -sync yes $waiting -j y -o /dev/null -e /dev/null -N $qsubname.W -b y /bin/ls 2>&1 > $qsubname.W.log
rm $qsubname.W.log

qsubout="$workingdir/SUBLM-OUT$$"
qsuberr="$workingdir/SUBLM-ERR$$"
qsublog="$workingdir/SUBLM-LOG$$"
qsubname="SUBLM"

unset getpids
echo "Estimating language models for each word list"

if [ $smoothing = "--kneser-ney" -o $smoothing = "--improved-kneser-ney" ]; then

for sfx in ${suffix[@]} ; do
(\
qsub $queueparameters -b no -j yes -sync no -o $qsubout.$sfx -e $qsuberr.$sfx -N $qsubname-$sfx << EOF
cd $workingdir
echo exit status $?

$scr/build-sublm.pl $verbose $prune $smoothing "cat $tmpdir/ikn.stat.dict*" --size $order --ngrams "$gunzip -c $tmpdir/ngram.dict.${sfx}.gz" -sublm $tmpdir/lm.dict.${sfx}  
echo exit status $?

echo
EOF
) 2>&1 > $qsublog.$sfx

id=`cat $qsublog.$sfx | grep 'Your job' | awk '{print $3}'`
sgepid[${#sgepid[@]}]=$id

done

else


for sfx in ${suffix[@]} ; do
(\
qsub $queueparameters -b no -j yes -sync no -o $qsubout.$sfx -e $qsuberr.$sfx -N $qsubname-$sfx << EOF
cd $workingdir
echo exit status $?

$scr/build-sublm.pl $verbose $prune $smoothing --size $order --ngrams "$gunzip -c $tmpdir/ngram.dict.${sfx}.gz" -sublm $tmpdir/lm.dict.${sfx}  

echo
EOF
) 2>&1 > $qsublog.$sfx

id=`cat $qsublog.$sfx | grep 'Your job' | awk '{print $3}'`
sgepid[${#sgepid[@]}]=$id

done

fi


waiting=""
for id in ${sgepid[@]} ; do waiting="$waiting -hold_jid $id" ; done


qsub $queueparameters -sync yes $waiting -o /dev/null -e /dev/null -N $qsubname.W -b yes /bin/ls 2>&1 > $qsubname.W.log
rm $qsubname.W.log

echo "Merging language models into $outfile"
qsubout="$workingdir/MERGE-OUT$$"
qsuberr="$workingdir/MERGE-ERR$$"
qsublog="$workingdir/MERGE-LOG$$"
qsubname="MERGE"
(\
qsub $queueparameters -b no -j yes -sync yes -o $qsubout -e $qsuberr -N $qsubname << EOF
cd $workingdir
$scr/merge-sublm.pl --size $order --sublm $tmpdir/lm.dict -lm $outfile
EOF
) 2>&1 > $qsublog

echo "Cleaning temporary directory $tmpdir";
rm $tmpdir/* 2> /dev/null
rm $qsubout* $qsuberr* $qsublog* 2> /dev/null

if [ $tmpdir_created -eq 1 ]; then
    echo "Removing temporary directory $tmpdir";
    rmdir $tmpdir 2> /dev/null
    if [ $? != 0 ]; then
        echo "Warning: the temporary directory could not be removed."
    fi
fi

exit 0

