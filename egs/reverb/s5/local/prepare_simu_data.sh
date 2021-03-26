#!/usr/bin/env bash
#
# Copyright 2018 Johns Hopkins University (Author: Shinji Watanabe)
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0
# This script is adapted from data preparation scripts in the Kaldi reverb recipe
# https://github.com/kaldi-asr/kaldi/tree/master/egs/reverb/s5/local

# Begin configuration section.
wavdir=${PWD}/wav
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <reverb-dir> <wsjcam0-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora5/REVERB_2014/REVERB /export/corpora3/LDC/LDC95S24/wsjcam0"
  exit 1
fi

set -e -o pipefail

reverb=$1
wsjcam0=$2

# tool directory
tooldir=${PWD}/data/local/reverb_tools

# working directory
dir=${PWD}/data/local/data
mkdir -p ${dir}

# make a one dot file for train, dev, and eval data
# the directory structure of WSJCAM0 is not consistent and we need such process for each task
cp ${wsjcam0}/data/primary_microphone/etc/si_tr.dot ${dir}/tr.dot
cat ${wsjcam0}/data/primary_microphone/etc/si_dt*.dot | sort > ${dir}/dt.dot
cat ${wsjcam0}/data/*/si_et*/*/*.dot | sort > ${dir}/et.dot

noiseword="<NOISE>";
for nch in 1 2 8; do
    taskdir=data/local/reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/${nch}ch
    # make a wav list
    task=tr
    for x in `ls ${taskdir} | grep SimData | grep _${task}_`; do
	perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${wavdir}/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
	    sed -e "s/^\(...\)/\1_${x}_\1/"
    done > ${dir}/${task}_simu_${nch}ch_wav.scp
    for task in dt et; do
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${reverb}/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch_wav.scp
	if [ ${nch} == 1 ]; then
	    for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e cln`; do
	        perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${reverb}/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
	    	sed -e "s/^\(...\)/\1_${x}_\1/"
	    done > ${dir}/${task}_cln_wav.scp
        fi
    done

    task=tr
    for x in `ls ${taskdir} | grep SimData | grep _${task}_`; do
	perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${wavdir}/WPE/${nch}ch/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
	    sed -e "s/^\(...\)/\1_${x}_\1/"
    done > ${dir}/${task}_simu_${nch}ch_wpe_wav.scp
    for task in dt et; do
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${wavdir}/WPE/${nch}ch/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch_wpe_wav.scp
    done

    # make a transcript
    task=tr
    for x in `ls ${taskdir} | grep SimData | grep _${task}_`; do
        perl -e 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, "\n"; } }' ${taskdir}/$x |\
	    perl local/find_transcripts_singledot.pl ${dir}/${task}.dot |\
	    sed -e "s/^\(...\)/\1_${x}_\1/"
    done > ${dir}/${task}_simu_${nch}ch.trans1 || exit 1;
    cat ${dir}/${task}_simu_${nch}ch.trans1 | local/normalize_transcript.pl ${noiseword} > ${dir}/${task}_simu_${nch}ch.txt || exit 1;
    for task in dt et; do
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -e 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, "\n"; } }' ${taskdir}/$x |\
		perl local/find_transcripts_singledot.pl ${dir}/${task}.dot |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch.trans1 || exit 1;
	cat ${dir}/${task}_simu_${nch}ch.trans1 | local/normalize_transcript.pl ${noiseword} > ${dir}/${task}_simu_${nch}ch.txt || exit 1;
	if [ ${nch} == 1 ]; then
	    for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e cln`; do
		perl -e 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, "\n"; } }' ${taskdir}/$x |\
		    perl local/find_transcripts_singledot.pl ${dir}/${task}.dot |\
		    sed -e "s/^\(...\)/\1_${x}_\1/"
	    done > ${dir}/${task}_cln.trans1 || exit 1;
	    cat ${dir}/${task}_cln.trans1 | local/normalize_transcript.pl ${noiseword} > ${dir}/${task}_cln.txt || exit 1;
        fi
    done
    
    # Make the utt2spk and spk2utt files.
    for task in tr dt et; do
	cat ${dir}/${task}_simu_${nch}ch_wav.scp | awk '{print $1}' | awk -F '_' '{print $0 " " $1}' > ${dir}/${task}_simu_${nch}ch.utt2spk || exit 1;
	cat ${dir}/${task}_simu_${nch}ch.utt2spk | ./utils/utt2spk_to_spk2utt.pl > ${dir}/${task}_simu_${nch}ch.spk2utt || exit 1;
    done
    for task in dt et; do
	cat ${dir}/${task}_cln_wav.scp | awk '{print $1}' | awk -F '_' '{print $0 " " $1}' > ${dir}/${task}_cln.utt2spk || exit 1;
	cat ${dir}/${task}_cln.utt2spk | ./utils/utt2spk_to_spk2utt.pl > ${dir}/${task}_cln.spk2utt || exit 1;
    done
done

# finally copy the above files to the data directory
for nch in 1 2 8; do
    for task in tr dt et; do
	datadir=data/${task}_simu_${nch}ch
	mkdir -p ${datadir}
	sort ${dir}/${task}_simu_${nch}ch_wav.scp > ${datadir}/wav.scp
	sort ${dir}/${task}_simu_${nch}ch.txt     > ${datadir}/text
	sort ${dir}/${task}_simu_${nch}ch.utt2spk > ${datadir}/utt2spk
	sort ${dir}/${task}_simu_${nch}ch.spk2utt > ${datadir}/spk2utt
	./utils/fix_data_dir.sh ${datadir}
	if [ ${task} != 'tr' ]; then
	    datadir=data/${task}_simu_${nch}ch_wpe
	    mkdir -p ${datadir}
	    sort ${dir}/${task}_simu_1ch_wpe_wav.scp | sed -e "s/WPE\/1ch/WPE\/${nch}ch/" > ${datadir}/wav.scp
	    sort ${dir}/${task}_simu_1ch.txt     > ${datadir}/text
	    sort ${dir}/${task}_simu_1ch.utt2spk > ${datadir}/utt2spk
	    sort ${dir}/${task}_simu_1ch.spk2utt > ${datadir}/spk2utt
	    ./utils/fix_data_dir.sh ${datadir}
	    if [ ${nch} != 1 ]; then
		datadir=data/${task}_simu_${nch}ch_beamformit
		mkdir -p ${datadir}
		sort ${dir}/${task}_simu_1ch_wpe_wav.scp | sed -e "s/ch1/bf${nch}/" | sed -e "s/WPE\/1ch/WPE\/${nch}ch/" > ${datadir}/wav.scp
		sort ${dir}/${task}_simu_1ch.txt     > ${datadir}/text
		sort ${dir}/${task}_simu_1ch.utt2spk > ${datadir}/utt2spk
		sort ${dir}/${task}_simu_1ch.spk2utt > ${datadir}/spk2utt
		./utils/fix_data_dir.sh ${datadir}
            else
		datadir=data/${task}_cln
		mkdir -p ${datadir}
		sort ${dir}/${task}_cln_wav.scp > ${datadir}/wav.scp
		sort ${dir}/${task}_cln.txt     > ${datadir}/text
		sort ${dir}/${task}_cln.utt2spk > ${datadir}/utt2spk
		sort ${dir}/${task}_cln.spk2utt > ${datadir}/spk2utt
		./utils/fix_data_dir.sh ${datadir}
	    fi
	fi
    done
done
