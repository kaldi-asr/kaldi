#!/bin/bash 

# begin configuration section.
min_lmwt=7
max_lmwt=17
stage=0
cer=0
ctm_name=
cmd=run.pl
#end configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

data=$1; 
q=$2; 
shift; shift;

if [ -z $ctm_name ] ; then
  ctm_name=`basename $data`;
fi

name=$ctm_name

for i in $@ ; do
    p=$q/`basename $i`
    [ ! -f $i/reco2file_and_channel ] && "The file reco2file_and_channel not present in the $i directory!" && exit 1
    for lmw in $q/score_* ; do
        d=$p/`basename $lmw`
        mkdir -p $d
        #echo " $lmw/$name.char.ctm "
        if [ $cer -eq 1 ] ; then
          [ !  -f $lmw/$name.char.ctm ] && echo "File $lmw/$name.char.ctm does not exist!" && exit 1
          utils/filter_scp.pl <(cut -f 1 -d ' ' $i/reco2file_and_channel) $lmw/$name.char.ctm > $d/`basename $i`.char.ctm
        fi

        [ ! -f $lmw/$name.ctm ] && echo "File $lmw/$name.ctm does not exist!" && exit 1
        utils/filter_scp.pl <(cut -f 1 -d ' ' $i/reco2file_and_channel) $lmw/$name.ctm > $d/`basename $i`.ctm
    done

    if [ -f $i/stm ] && [ -f $i/glm ]; then
        local/score_stm.sh --min-lmwt $min_lmwt --max-lmwt $max_lmwt --cer $cer --cmd "$cmd" $i data/lang $p
    else
        echo "Not running scoring, file $i/stm does not exist"
    fi

done
exit 0

