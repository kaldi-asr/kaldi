#!/bin/bash

data=$1; 
q=$2; 
shift; shift;

name=`basename $data`;

for i in $@ ; do
    p=$q/`basename $i`
    [ ! -f $i/reco2file_and_channel ] && "The file reco2file_and_channel not present in the $i directory!" && exit 1
    for lmw in $q/score_* ; do
        d=$p/`basename $lmw`
        mkdir -p $d
        #echo " $lmw/$name.char.ctm "
        [ -f $lmw/$name.char.ctm ] && \
          utils/filter_scp.pl <(cut -f 1 -d ' ' $i/reco2file_and_channel) $lmw/$name.char.ctm > $d/`basename $i`.char.ctm
        [ -f $lmw/$name.ctm ] && \
          utils/filter_scp.pl <(cut -f 1 -d ' ' $i/reco2file_and_channel) $lmw/$name.ctm > $d/`basename $i`.ctm
    done

    if [ -f $i/stm ] && [ -f $i/glm ]; then
        local/score_scm.sh --cmd "$decode_cmd" $i data/lang $p
    else
        echo "Not running scoring, file $i/stm does not exist"
    fi

done
exit 0

