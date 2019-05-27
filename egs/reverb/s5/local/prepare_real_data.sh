#!/bin/bash
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
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <reverb-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora5/REVERB_2014/REVERB"
  exit 1
fi

set -e -o pipefail

reverb=$1

# working directory
dir=${PWD}/data/local/data
mkdir -p ${dir}

for task in dt et; do
    if [ ${task} == 'dt' ]; then
	mlf=${reverb}/MC_WSJ_AV_Dev/mlf/WSJ.mlf
    elif [ ${task} == 'et' ]; then
	mlf=${reverb}/MC_WSJ_AV_Eval/mlf/WSJ.mlf
    fi
    # MLF transcription correction
    # taken from HTK baseline script
    sed -e '
# dos to unix line feed conversion
s/\x0D$//' \
	-e "
            s/\x60//g              # remove unicode character grave accent.
       " \
	-e "
            # fix the single quote for the word yield
            # and the quoted ROOTS
            # e.g. yield' --> yield
            # reason: YIELD' is not in dict, while YIELD is
            s/YIELD'/YIELD/g
            s/'ROOTS'/ROOTS/g
            s/'WHERE/WHERE/g
            s/PEOPLE'/PEOPLE/g
            s/SIT'/SIT/g
            s/'DOMINEE/DOMINEE/g
            s/CHURCH'/CHURCH/g" \
	-e '
              # fix the single missing double full stop issue at the end of an utterance
              # e.g. I. C. N should be  I. C. N.
              # reason: N is not in dict, while N. is
              /^[A-Z]$/ {
              # append a line
                      N
              # search for single dot on the second line
                      /\n\./ {
              # found it - now replace the
                              s/\([A-Z]\)\n\./\1\.\n\./
                      }
              }' \
	$mlf |\
	perl local/mlf2text.pl > ${dir}/${task}.txt
done


noiseword="<NOISE>";
for nch in 1 2 8; do
    taskdir=data/local/reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/${nch}ch
    # make a wav list
    for task in dt et; do
	if [ ${task} == 'dt' ]; then
	    audiodir=${reverb}/MC_WSJ_AV_Dev
	    audiodir_wpe=${wavdir}/WPE/${nch}ch/MC_WSJ_AV_Dev
	elif [ ${task} == 'et' ]; then
	    audiodir=${reverb}/MC_WSJ_AV_Eval
	    audiodir_wpe=${wavdir}/WPE/${nch}ch/MC_WSJ_AV_Eval
	fi
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id $dir$_";}' -- -dir=${audiodir} ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch_wav.scp
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id $dir$_";}' -- -dir=${audiodir_wpe} ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch_wpe_wav.scp
    done
    # make a transcript
    for task in dt et; do
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id\n";}' ${taskdir}/$x |\
		perl local/find_transcripts_txt.pl ${dir}/${task}.txt |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch.trans1 || exit 1;
	cat ${dir}/${task}_real_${nch}ch.trans1 | local/normalize_transcript.pl ${noiseword} > ${dir}/${task}_real_${nch}ch.txt || exit 1;
    done
    
    # Make the utt2spk and spk2utt files.
    for task in dt et; do
	cat ${dir}/${task}_real_${nch}ch_wav.scp | awk '{print $1}' | awk -F '_' '{print $0 " " $1}' > ${dir}/${task}_real_${nch}ch.utt2spk || exit 1;
	cat ${dir}/${task}_real_${nch}ch.utt2spk | ./utils/utt2spk_to_spk2utt.pl > ${dir}/${task}_real_${nch}ch.spk2utt || exit 1;
    done
done

# finally copy the above files to the data directory
for nch in 1 2 8; do
    for task in dt et; do
	datadir=data/${task}_real_${nch}ch
	mkdir -p ${datadir}
	sort ${dir}/${task}_real_${nch}ch_wav.scp > ${datadir}/wav.scp
	sort ${dir}/${task}_real_${nch}ch.txt     > ${datadir}/text
	sort ${dir}/${task}_real_${nch}ch.utt2spk > ${datadir}/utt2spk
	sort ${dir}/${task}_real_${nch}ch.spk2utt > ${datadir}/spk2utt
	./utils/fix_data_dir.sh ${datadir}
	if [ ${nch} != 1 ]; then
	    datadir=data/${task}_real_${nch}ch_beamformit
	    mkdir -p ${datadir}
	    sort ${dir}/${task}_real_1ch_wpe_wav.scp | sed -e "s/-[1-8]_/-bf${nch}_/" | sed -e "s/WPE\/1ch/WPE\/${nch}ch/" > ${datadir}/wav.scp
	    sort ${dir}/${task}_real_1ch.txt     > ${datadir}/text
	    sort ${dir}/${task}_real_1ch.utt2spk > ${datadir}/utt2spk
	    sort ${dir}/${task}_real_1ch.spk2utt > ${datadir}/spk2utt
	    ./utils/fix_data_dir.sh ${datadir}
	fi
	datadir=data/${task}_real_${nch}ch_wpe
	mkdir -p ${datadir}
	sort ${dir}/${task}_real_1ch_wpe_wav.scp | sed -e "s/WPE\/1ch/WPE\/${nch}ch/" > ${datadir}/wav.scp
	sort ${dir}/${task}_real_1ch.txt     > ${datadir}/text
	sort ${dir}/${task}_real_1ch.utt2spk > ${datadir}/utt2spk
	sort ${dir}/${task}_real_1ch.spk2utt > ${datadir}/spk2utt
	./utils/fix_data_dir.sh ${datadir}
    done
done
