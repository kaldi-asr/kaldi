#!/bin/bash
# Generates mcep from a list of wav file.

#export PATH=/home/potard/aria/idlak/src/featbin:$PATH

export tooldir=$KALDI_ROOT/tools/SPTK/bin

help_message="Usage: ./compute-mcep-feats.sh [options] scp:<in.scp> <wspecifier>\n\tcf. top of file for list of options."

fshift=5
srate=48000
order=60
alpha=0.55
frame_length=25
extra_opts=""

#echo "$0 $@"  # Print the command line for logging

. parse_options.sh

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: compute-mcep-feats.sh [options] scp:<in.scp> <wspecifier>"
   echo " => will generate mcep using SPTK mcep tool"
   echo " e.g.: compute-mcep-feats.sh wav.scp ark:feats.ark"
   exit 1;
fi

mlen=$(( $order + 1 ))
winstep=$(( $srate * $fshift / 1000 ))
winlen=$(( $srate * $frame_length / 1000 ))
fftlen=`awk -v len=$winlen 'BEGIN{print 2**(1+int(log(len-0.1)/log(2)))}'`
echo "Running with $winstep, $winlen, $fftlen" 1>&2

for i in `awk -v lst="$1" 'BEGIN{if (lst ~ /^scp/) sub("[^:]+:[[:space:]]*","", lst); while (getline < lst) print $1 "___" $2}'`; do
    name=${i%%___*}
    wfilename=${i##*___}
    #echo $name $wfilename
    
    sox $wfilename -t raw - | x2x +sf | $tooldir/frame -l $winlen -p $winstep \
	| $tooldir/window -l $winlen -L $fftlen \
	| $tooldir/mcep -e 1.0e-8 -f 0.0 -j 0 -l $fftlen -m $order -a $alpha \
	| $tooldir/x2x +f +a$mlen \
	| awk -v name=$name '
BEGIN{print name, "[";}
{print}
END{print "]"}'
done | copy-feats ark:- "$2"

#-j 100 -f 1e-12
# -d 1e-8 -e 1.0e-8   -j 100 -f 0.0
