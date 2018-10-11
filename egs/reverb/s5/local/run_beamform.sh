#!/bin/bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)
# Copyright 2018, Johns Hopkins University (Author: Aswin Shanmugam Subramanian)

. ./cmd.sh
. ./path.sh

# Config:
nj=20
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Wrong #arguments ($#, expected 1)"
   echo "Usage: local/run_beamform.sh [options] <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

odir=$1
dir=${PWD}/data/local/data

if [ -z $BEAMFORMIT ] ; then
  export BEAMFORMIT=$KALDI_ROOT/tools/extras/BeamformIt
fi
export PATH=${PATH}:$BEAMFORMIT
! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; extras/install_beamformit.sh;'" && exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

for task in dt et; do
    for nch in 2 8; do
        wdir=exp/beamform_real_${task}_${nch}ch
        mkdir -p $wdir/log
        arrays=$wdir/channels
        output_wavfiles=$wdir/wavfiles.list
        if [ ${nch} == 2 ]; then
            allwavs=`cat ${dir}/${task}_real_${nch}ch_wpe_wav.scp | cut -d " " -f2`
            allwavs_beamformit=`cat data/${task}_real_${nch}ch_beamformit/wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==1' > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==0' > $wdir/channels.2nd
            echo $allwavs_beamformit | tr ' ' '\n' | rev | sort | rev | awk -F 'WPE/' '{print $2}' | awk -F '.wav' '{print $1}' > $output_wavfiles
            paste -d" " $output_wavfiles $wdir/channels.1st $wdir/channels.2nd > $arrays
        elif [ ${nch} == 8 ]; then
            allwavs=`cat ${dir}/${task}_real_${nch}ch_wpe_wav.scp | cut -d " " -f2`
            allwavs_beamformit=`cat data/${task}_real_${nch}ch_beamformit/wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==1' > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==2' > $wdir/channels.2nd
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==3' > $wdir/channels.3rd
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==4' > $wdir/channels.4th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==5' > $wdir/channels.5th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==6' > $wdir/channels.6th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==7' > $wdir/channels.7th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==0' > $wdir/channels.8th
            echo $allwavs_beamformit | tr ' ' '\n' | rev | sort | rev | awk -F 'WPE/' '{print $2}' | awk -F '.wav' '{print $1}' > $output_wavfiles
            paste -d" " $output_wavfiles $wdir/channels.1st $wdir/channels.2nd $wdir/channels.3rd $wdir/channels.4th $wdir/channels.5th $wdir/channels.6th $wdir/channels.7th $wdir/channels.8th > $arrays
        fi
	# split the list for parallel processing
        split_wavfiles=""
        for n in `seq $nj`; do
          split_wavfiles="$split_wavfiles $output_wavfiles.$n"
        done
        utils/split_scp.pl $output_wavfiles $split_wavfiles || exit 1;
        
        echo -e "Beamforming - $task - real - $nch ch\n"
        # making a shell script for each job
        for n in `seq $nj`; do
	cat <<-EOF > $wdir/log/beamform.$n.sh
	while read line; do
	  $BEAMFORMIT/BeamformIt -s \$line -c $arrays \
	    --config_file `pwd`/conf/reverb_beamformit.cfg \
	    --result_dir $odir
	done < $output_wavfiles.$n
	EOF
        done
        
        chmod a+x $wdir/log/beamform.*.sh
        $cmd JOB=1:$nj $wdir/log/beamform.JOB.log \
          $wdir/log/beamform.JOB.sh
    done
done

for task in dt et; do
    for nch in 2 8; do
        wdir=exp/beamform_simu_${task}_${nch}ch
        mkdir -p $wdir/log
        arrays=$wdir/channels
        output_wavfiles=$wdir/wavfiles.list
        if [ ${nch} == 2 ]; then
            allwavs=`cat ${dir}/${task}_simu_${nch}ch_wpe_wav.scp | grep "ch[1-2].wav" | cut -d " " -f2`
            allwavs_beamformit=`cat data/${task}_simu_${nch}ch_beamformit/wav.scp | grep "bf2.wav" | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels.2nd
            echo $allwavs_beamformit | tr ' ' '\n' | awk -F 'WPE/' '{print $2}' | sort | awk -F '.wav' '{print $1}'  > $output_wavfiles
            paste -d" " $output_wavfiles $wdir/channels.1st $wdir/channels.2nd > $arrays
        elif [ ${nch} == 8 ]; then
            allwavs=`cat ${dir}/${task}_simu_${nch}ch_wpe_wav.scp | grep "ch[1-8].wav" | cut -d " " -f2`
            allwavs_beamformit=`cat data/${task}_simu_${nch}ch_beamformit/wav.scp | grep "bf8.wav" | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels.2nd
            echo $allwavs | tr ' ' '\n' | grep 'ch3' | sort > $wdir/channels.3rd
            echo $allwavs | tr ' ' '\n' | grep 'ch4' | sort > $wdir/channels.4th
            echo $allwavs | tr ' ' '\n' | grep 'ch5' | sort > $wdir/channels.5th
            echo $allwavs | tr ' ' '\n' | grep 'ch6' | sort > $wdir/channels.6th
            echo $allwavs | tr ' ' '\n' | grep 'ch7' | sort > $wdir/channels.7th
            echo $allwavs | tr ' ' '\n' | grep 'ch8' | sort > $wdir/channels.8th
            echo $allwavs_beamformit | tr ' ' '\n' | awk -F 'WPE/' '{print $2}' | sort | awk -F '.wav' '{print $1}' > $output_wavfiles
            paste -d" " $output_wavfiles $wdir/channels.1st $wdir/channels.2nd $wdir/channels.3rd $wdir/channels.4th $wdir/channels.5th $wdir/channels.6th $wdir/channels.7th $wdir/channels.8th > $arrays
        fi
	# split the list for parallel processing
        split_wavfiles=""
        for n in `seq $nj`; do
          split_wavfiles="$split_wavfiles $output_wavfiles.$n"
        done
        utils/split_scp.pl $output_wavfiles $split_wavfiles || exit 1;
        
        echo -e "Beamforming - $task - simu - $nch ch\n"
        # making a shell script for each job
        for n in `seq $nj`; do
	cat <<-EOF > $wdir/log/beamform.$n.sh
	while read line; do
	  $BEAMFORMIT/BeamformIt -s \$line -c $arrays \
	    --config_file `pwd`/conf/reverb_beamformit.cfg \
	    --result_dir $odir
	done < $output_wavfiles.$n
	EOF
        done
        
        chmod a+x $wdir/log/beamform.*.sh
        $cmd JOB=1:$nj $wdir/log/beamform.JOB.log \
          $wdir/log/beamform.JOB.sh
    done
done
echo "`basename $0` Done."
