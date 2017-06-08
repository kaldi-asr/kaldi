#! /bin/bash

function usage()
{
    cmnd=$(basename $0);
    cat<<EOF

$cmnd - creates partition files with ngram statistics in Google format

USAGE:
       $cmnd [options] <input> <output> <order> <parts>

DESCRIPTION:
       <input>   Input file name
       <output>  Partition files name prefix
       <order>   Order of the ngrams
       <parts>   Number of partitions

OPTIONS:
       -h        Show this message

EOF
}

# Parse options
while getopts h OPT; do
    case "$OPT" in
        h)
            usage >&2;
            exit 0;
            ;;
        * ) usage;
            exit 1;
						;;
    esac
done

#usage:
#ngt-split.sh [options] <input> <output> <size> <parts>
#It creates <parts> files (named <output.000>, ... <output.999>)
#containing ngram statistics (of <order> length) in Google format
#These files are a partition of the whole set of ngrams

basedir=$IRSTLM
bindir=$basedir/bin
scriptdir=$basedir/scripts

unset par
while [ $# -gt 0 ]
do
   echo "$0: arg $1"
   par[${#par[@]}]="$1"
   shift
done

inputfile=${par[0]}
outputfile=${par[1]}
order=${par[2]}
parts=${par[3]}

dictfile=dict$$


echo "Extracting dictionary from training corpus"
$bindir/dict -i="$inputfile" -o=$dictfile -f=y -sort=n

echo "Splitting dictionary into $parts lists"
$scriptdir/split-dict.pl --input $dictfile --output ${dictfile}. --parts $parts

rm $dictfile


echo "Extracting n-gram statistics for each word list"
echo "Important: dictionary must be ordered according to order of appearance of words in data"
echo "used to generate n-gram blocks,  so that sub language model blocks results ordered too"

for d in `ls ${dictfile}.*` ; do
w=`echo $d | perl -pe 's/.+(\.[0-9]+)$/$1/i'`
w="$outputfile$w"

sdict=`basename $sdict`
echo "Extracting n-gram statistics for $sdict"

echo "$bindir/ngt -i="$inputfile"  -n=$order -gooout=y -o=$w -fd=$d  > /dev/null"
$bindir/ngt -n=$order -gooout=y -o=$w -fd=$d -i="$inputfile"  > /dev/null
rm $d
done

exit 0
