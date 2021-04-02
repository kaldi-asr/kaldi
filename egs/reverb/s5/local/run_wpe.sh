#!/usr/bin/env bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:
nj=50
cmd=run.pl

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run '../../../tools/extras/install_miniconda.sh' and '../../../tools/extras/install_wpe.sh';"
fi

# check if WPE is installed
result=`$HOME/miniconda3/bin/python -c "\
try:
    import nara_wpe
    print('1')
except ImportError:
    print('0')"`

if [ "$result" == "1" ]; then
    echo "WPE is installed"
else
    echo "WPE is not installed. Please run ../../../tools/extras/install_wpe.sh"
fi

dir=${PWD}/data/local/data

for task in dt et; do
    for nch in 1 2 8; do
        wdir=exp/wpe_real_${task}_${nch}ch
        mkdir -p $wdir/log
        arrays=$wdir/channels
        output_wavfiles=$wdir/wavfiles.list
        if [ ${nch} == 1 ]; then
            allwavs=`cat ${dir}/${task}_real_1ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_real_1ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' > $wdir/channels_input
            echo $allwavs_output | tr ' ' '\n' > $wdir/channels_output
            paste -d" " $wdir/channels_input $wdir/channels_output > $arrays
        elif [ ${nch} == 2 ]; then
            allwavs=`cat ${dir}/${task}_real_2ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_real_2ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==1' > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==0' > $wdir/channels.2nd
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==1' > $wdir/channels_output.1st
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%2==0' > $wdir/channels_output.2nd
            paste -d" " $wdir/channels.1st $wdir/channels.2nd $wdir/channels_output.1st $wdir/channels_output.2nd > $arrays
        elif [ ${nch} == 8 ]; then
            allwavs=`cat ${dir}/${task}_real_8ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_real_8ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==1' > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==2' > $wdir/channels.2nd
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==3' > $wdir/channels.3rd
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==4' > $wdir/channels.4th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==5' > $wdir/channels.5th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==6' > $wdir/channels.6th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==7' > $wdir/channels.7th
            echo $allwavs | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==0' > $wdir/channels.8th
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==1' > $wdir/channels_output.1st
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==2' > $wdir/channels_output.2nd
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==3' > $wdir/channels_output.3rd
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==4' > $wdir/channels_output.4th
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==5' > $wdir/channels_output.5th
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==6' > $wdir/channels_output.6th
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==7' > $wdir/channels_output.7th
            echo $allwavs_output | tr ' ' '\n' | rev | sort | rev | awk 'NR%8==0' > $wdir/channels_output.8th
            paste -d" " $wdir/channels.1st $wdir/channels.2nd $wdir/channels.3rd $wdir/channels.4th $wdir/channels.5th $wdir/channels.6th $wdir/channels.7th $wdir/channels.8th $wdir/channels_output.1st $wdir/channels_output.2nd $wdir/channels_output.3rd $wdir/channels_output.4th $wdir/channels_output.5th $wdir/channels_output.6th $wdir/channels_output.7th $wdir/channels_output.8th > $arrays
        fi
        
        # split the list for parallel processing
        split_wavfiles=""
        for n in `seq $nj`; do
            split_wavfiles="$split_wavfiles $output_wavfiles.$n"
        done
        utils/split_scp.pl $arrays $split_wavfiles || exit 1;
        
        echo -e "Dereverberation - $task - real - $nch ch\n"
        # making a shell script for each job
	for n in `seq $nj`; do
	cat <<-EOF > $wdir/log/wpe.$n.sh
	while read line; do
	  $HOME/miniconda3/bin/python local/run_wpe.py \
	    --file \$line
	done < $output_wavfiles.$n
	EOF
	done

        chmod a+x $wdir/log/wpe.*.sh
        $cmd JOB=1:$nj $wdir/log/wpe.JOB.log \
          $wdir/log/wpe.JOB.sh
    done
done

for task in dt et; do
    for nch in 1 2 8; do
        wdir=exp/wpe_simu_${task}_${nch}ch
        mkdir -p $wdir/log
        arrays=$wdir/channels
        output_wavfiles=$wdir/wavfiles.list
        if [ ${nch} == 1 ]; then
            allwavs=`cat ${dir}/${task}_simu_1ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_simu_1ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' > $wdir/channels_input
            echo $allwavs_output | tr ' ' '\n' > $wdir/channels_output
            paste -d" " $wdir/channels_input $wdir/channels_output > $arrays
        elif [ ${nch} == 2 ]; then
            allwavs=`cat ${dir}/${task}_simu_2ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_simu_2ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels.2nd
            echo $allwavs_output | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels_output.1st
            echo $allwavs_output | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels_output.2nd
            paste -d" " $wdir/channels.1st $wdir/channels.2nd $wdir/channels_output.1st $wdir/channels_output.2nd > $arrays
        elif [ ${nch} == 8 ]; then
            allwavs=`cat ${dir}/${task}_simu_8ch_wav.scp | cut -d " " -f2`
            allwavs_output=`cat ${dir}/${task}_simu_8ch_wpe_wav.scp | cut -d " " -f2`
            echo $allwavs | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels.1st
            echo $allwavs | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels.2nd
            echo $allwavs | tr ' ' '\n' | grep 'ch3' | sort > $wdir/channels.3rd
            echo $allwavs | tr ' ' '\n' | grep 'ch4' | sort > $wdir/channels.4th
            echo $allwavs | tr ' ' '\n' | grep 'ch5' | sort > $wdir/channels.5th
            echo $allwavs | tr ' ' '\n' | grep 'ch6' | sort > $wdir/channels.6th
            echo $allwavs | tr ' ' '\n' | grep 'ch7' | sort > $wdir/channels.7th
            echo $allwavs | tr ' ' '\n' | grep 'ch8' | sort > $wdir/channels.8th
            echo $allwavs_output | tr ' ' '\n' | grep 'ch1' | sort > $wdir/channels_output.1st
            echo $allwavs_output | tr ' ' '\n' | grep 'ch2' | sort > $wdir/channels_output.2nd
            echo $allwavs_output | tr ' ' '\n' | grep 'ch3' | sort > $wdir/channels_output.3rd
            echo $allwavs_output | tr ' ' '\n' | grep 'ch4' | sort > $wdir/channels_output.4th
            echo $allwavs_output | tr ' ' '\n' | grep 'ch5' | sort > $wdir/channels_output.5th
            echo $allwavs_output | tr ' ' '\n' | grep 'ch6' | sort > $wdir/channels_output.6th
            echo $allwavs_output | tr ' ' '\n' | grep 'ch7' | sort > $wdir/channels_output.7th
            echo $allwavs_output | tr ' ' '\n' | grep 'ch8' | sort > $wdir/channels_output.8th
            paste -d" " $wdir/channels.1st $wdir/channels.2nd $wdir/channels.3rd $wdir/channels.4th $wdir/channels.5th $wdir/channels.6th $wdir/channels.7th $wdir/channels.8th $wdir/channels_output.1st $wdir/channels_output.2nd $wdir/channels_output.3rd $wdir/channels_output.4th $wdir/channels_output.5th $wdir/channels_output.6th $wdir/channels_output.7th $wdir/channels_output.8th > $arrays
        fi
        
        # split the list for parallel processing
        split_wavfiles=""
        for n in `seq $nj`; do
            split_wavfiles="$split_wavfiles $output_wavfiles.$n"
        done
        utils/split_scp.pl $arrays $split_wavfiles || exit 1;
        
        echo -e "Dereverberation - $task - simu - $nch ch\n"
        # making a shell script for each job
	for n in `seq $nj`; do
	cat <<-EOF > $wdir/log/wpe.$n.sh
	while read line; do
	  $HOME/miniconda3/bin/python local/run_wpe.py \
	    --file \$line
	done < $output_wavfiles.$n
	EOF
	done

        chmod a+x $wdir/log/wpe.*.sh
        $cmd JOB=1:$nj $wdir/log/wpe.JOB.log \
          $wdir/log/wpe.JOB.sh
    done
done
echo "`basename $0` Done."
