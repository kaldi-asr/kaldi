#! /bin/bash

set -m # Enable Job Control

function usage()
{
    cmnd=$(basename $0);
    cat<<EOF

$cmnd - estimates a language model file and saves it in intermediate ARPA format

USAGE:
       $cmnd [options]

OPTIONS:
       -i|--InputFile          Input training file e.g. 'gunzip -c train.gz'
       -o|--OutputFile         Output gzipped LM, e.g. lm.gz
       -k|--Parts              Number of splits (default 5)
       -n|--NgramSize          Order of language model (default 3)
       -d|--Dictionary         Define subdictionary for n-grams (optional, default is without any subdictionary)
       -s|--LanguageModelType  Smoothing methods: witten-bell (default), shift-beta, improved-shift-beta, stupid-backoff; kneser-ney and improved-kneser-ney still accepted for back-compatibility, but mapped into shift-beta and improved-shift-beta, respectively
       -p|--PruneSingletons    Prune singleton n-grams (default false)
       -f|--PruneFrequencyThreshold      Pruning frequency threshold for each level; comma-separated list of values; (default is '0,0,...,0', for all levels)
       -t|--TmpDir             Directory for temporary files (default ./stat_PID)
       -l|--LogFile            File to store logging info (default /dev/null)
       -u|--uniform            Use uniform word frequency for dictionary splitting (default false)
       -b|--boundaries         Include sentence boundary n-grams (optional, default false)
       -v|--verbose            Verbose
       --debug                 Debug
       -h|-?|--help            Show this message

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

#check irstlm installation
if [ ! -e $bin/dict -o  ! -e $scr/split-dict.pl ]; then
   echo "$IRSTLM does not contain a proper installation of IRSTLM"
   exit 3
fi

#default parameters
logfile=/dev/null
tmpdir=stat_$$
order=3
parts=3
inpfile="";
outfile=""
verbose="";
debug="";
smoothing="witten-bell";
prune="";
prune_thr_str="";
boundaries="";
dictionary="";
uniform="-f=y";
backoff=""

while [ "$1" != "" ]; do
    case $1 in
        -i | --InputFile )          shift;
																inpfile=$1;
																;;
        -o | --OutputFile )         shift;
																outfile=$1;
                                ;;
        -n | --NgramSize )           shift;
																order=$1;
                                ;;
        -k | --Parts )          shift;
																parts=$1;
                                ;;
        -d | --Dictionary )     shift;
                                dictionary="-sd=$1";
                                ;;
        -s | --LanguageModelType )        shift;
																				  smoothing=$1;
                                          ;;
        -f | --PruneFrequencyThreshold )  shift;
																          prune_thr_str="--PruneFrequencyThreshold=$1";
                                          ;;
        -p | --PruneSingletons )     prune='--prune-singletons';
																			;;
        -l | --LogFile )        shift;
																logfile=$1;
                                ;;
        -t | --TmpDir )         shift;
																tmpdir=$1;
                                ;;
        -u | --uniform )        uniform=' ';
                                ;;
        -b | --boundaries )     boundaries='--cross-sentence';
																;;
        -v | --verbose )        verbose='--verbose';
                                ;;
        --debug )               debug='--debug';
                                ;;
        -h | -? | --help )      usage;
                                exit 0;
                                ;;
        * )                     usage;
                                exit 1;
    esac
    shift
done

case $smoothing in
witten-bell) 
smoothing="--witten-bell";
;; 
kneser-ney)
## kneser-ney still accepted for back-compatibility, but mapped into shift-beta
smoothing="--shift-beta";
;;
improved-kneser-ney)
## improved-kneser-ney still accepted for back-compatibility, but mapped into improved-shift-beta
smoothing="--improved-shift-beta"; 
;;
shift-beta)
smoothing="--shift-beta";
;;
improved-shift-beta)
smoothing="--improved-shift-beta";
;;
stupid-backoff)
smoothing="--stupid-backoff";
backoff="--backoff"
;;
*) 
echo "wrong smoothing setting; '$smoothing' does not exist";
exit 4
esac
			

echo "LOGFILE:$logfile"
			 

if [ $verbose ] ; then
echo inpfile='"'$inpfile'"' outfile=$outfile order=$order parts=$parts tmpdir=$tmpdir prune=$prune smoothing=$smoothing dictionary=$dictionary verbose=$verbose debug=$debug prune_thr_str=$prune_thr_str  >> $logfile 2>&1
fi

if [ ! "$inpfile" -o ! "$outfile" ] ; then
    usage
    exit 5
fi
 
if [ -e $outfile ]; then
   echo "Output file $outfile already exists! either remove or rename it."
   exit 6
fi

if [ -e $logfile -a $logfile != "/dev/null" -a $logfile != "/dev/stdout" ]; then
   echo "Logfile $logfile already exists! either remove or rename it."
   exit 7
fi

echo "BIS LOGFILE:$logfile" >> $logfile 2>&1

#check tmpdir
tmpdir_created=0;
if [ ! -d $tmpdir ]; then
   echo "Temporary directory $tmpdir does not exist"  >> $logfile 2>&1
   echo "creating $tmpdir"  >> $logfile 2>&1
   mkdir -p $tmpdir
   tmpdir_created=1
else
   echo "Cleaning temporary directory $tmpdir" >> $logfile 2>&1
    rm $tmpdir/* 2> /dev/null
    if [ $? != 0 ]; then
        echo "Warning: some temporary files could not be removed" >> $logfile 2>&1
    fi
fi


echo "Extracting dictionary from training corpus" >> $logfile 2>&1
$bin/dict -i="$inpfile" -o=$tmpdir/dictionary $uniform -sort=no 2> $logfile

echo "Splitting dictionary into $parts lists" >> $logfile 2>&1
$scr/split-dict.pl --input $tmpdir/dictionary --output $tmpdir/dict. --parts $parts >> $logfile 2>&1

echo "Extracting n-gram statistics for each word list" >> $logfile 2>&1
echo "Important: dictionary must be ordered according to order of appearance of words in data" >> $logfile 2>&1
echo "used to generate n-gram blocks,  so that sub language model blocks results ordered too" >> $logfile 2>&1

for sdict in $tmpdir/dict.*;do
sdict=`basename $sdict`
echo "Extracting n-gram statistics for $sdict" >> $logfile 2>&1
if [ $smoothing = "--shift-beta" -o $smoothing = "--improved-shift-beta" ]; then
additional_parameters="-iknstat=$tmpdir/ikn.stat.$sdict"
else
additional_parameters=""
fi

$bin/ngt -i="$inpfile" -n=$order -gooout=y -o="$gzip -c > $tmpdir/ngram.${sdict}.gz" -fd="$tmpdir/$sdict" $dictionary $additional_parameters >> $logfile 2>&1 &

#$bin/ngt -i="$inpfile" -n=$order -gooout=y -o="$gzip -c > $tmpdir/ngram.${sdict}.gz" -fd="$tmpdir/$sdict" $dictionary -iknstat="$tmpdir/ikn.stat.$sdict" >> $logfile 2>&1 &
#else
#$bin/ngt -i="$inpfile" -n=$order -gooout=y -o="$gzip -c > $tmpdir/ngram.${sdict}.gz" -fd="$tmpdir/$sdict" $dictionary >> $logfile 2>&1 &
#fi
done

# Wait for all parallel jobs to finish
while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done

echo "Estimating language models for each word list" >> $logfile 2>&1
for sdict in `ls $tmpdir/dict.*` ; do
sdict=`basename $sdict`
echo "Estimating language models for $sdict" >> $logfile 2>&1

if [ $smoothing = "--shift-beta" -o $smoothing = "--improved-shift-beta" ]; then
additional_smoothing_parameters="cat $tmpdir/ikn.stat.dict.*"
additional_parameters="$backoff"
else
additional_smoothing_parameters=""
additional_parameters=""
fi
$scr/build-sublm.pl $verbose $prune $prune_thr_str $smoothing "$additional_smoothing_parameters" --size $order --ngrams "$gunzip -c $tmpdir/ngram.${sdict}.gz" -sublm $tmpdir/lm.$sdict $additional_parameters >> $logfile 2>&1 &

#if [ $smoothing = "--shift-beta" -o $smoothing = "--improved-shift-beta" ]; then
#$scr/build-sublm.pl $verbose $prune $prune_thr_str $smoothing "cat $tmpdir/ikn.stat.dict.*" --size $order --ngrams "$gunzip -c $tmpdir/ngram.${sdict}.gz" -sublm $tmpdir/lm.$sdict $backoff >> $logfile 2>&1 &
#else
#$scr/build-sublm.pl $verbose $prune $prune_thr_str $smoothing --size $order --ngrams "$gunzip -c $tmpdir/ngram.${sdict}.gz" -sublm $tmpdir/lm.$sdict >> $logfile 2>&1 &
#fi

done

# Wait for all parallel jobs to finish
while [ 1 ]; do fg 2> /dev/null; [ $? == 1 ] && break; done

echo "Merging language models into $outfile" >> $logfile 2>&1
$scr/merge-sublm.pl --size $order --sublm $tmpdir/lm.dict -lm $outfile $backoff  >> $logfile 2>&1

if [ $debug == "--debug" ] ; then
    echo "Debugging is active; hence, not removing temporary directory $tmpdir" >> $logfile 2>&1
exit 0
fi

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




