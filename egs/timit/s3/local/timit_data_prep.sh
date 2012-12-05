#!/bin/bash -u

# Copyright 2012  Navdeep Jaitly
# Copyright 2010-2011  Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from one directory above this script.

# The input is the 3 CDs from the LDC distribution of Resource Management.
# The script's argument is a directory which has three subdirectories:
# rm1_audio1  rm1_audio2  rm2_audio

# Note: when creating your own data preparation scripts, it's a good idea
# to make sure that the speaker id (if present) is a prefix of the utterance
# id, that the output scp file is sorted on utterance id, and that the 
# transcription file is exactly the same length as the scp file and is also
# sorted on utterance id (missing transcriptions should be removed from the
# scp file using e.g. scripts/filter_scp.pl)

if [ $# != 1 ]; then
  echo "Usage: ../../local/timit_data_prep.sh /path/to/TIMIT"
  exit 1; 
fi 

TIMIT_ROOT=$1
S3_ROOT=`pwd`
mkdir -p data/local
cd data/local

lower_case=0
upper_case=0
if [ -d $TIMIT_ROOT/TIMIT/TRAIN -a -d $TIMIT_ROOT/TIMIT/TEST ];
 then
   upper_case=1
   train_folder=$TIMIT_ROOT/TIMIT/TRAIN
   test_folder=$TIMIT_ROOT/TIMIT/TEST
   spkr_info_file=$TIMIT_ROOT/TIMIT/DOC/SPKRINFO.TXT
elif [ -d $TIMIT_ROOT/timit/train -a -d $TIMIT_ROOT/timit/test ];
 then
   lower_case=1
   train_folder=$TIMIT_ROOT/timit/train
   test_folder=$TIMIT_ROOT/timit/test
   spkr_info_file=$TIMIT_ROOT/timit/doc/spkrinfo.txt
else 
   echo "Error: run.sh requires a directory argument (an absolute pathname) that contains TIMIT/TRAIN and TIMIT/TEST or timit/train and timit/test."
   exit 1;
fi


(
   find $train_folder -iname "*.wav" | perl -ane 'if (! m/sa[0-9].wav/i){ print $_ ; }'
)  > train_sph.flist


# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
$S3_ROOT/local/make_trans.pl trn train_sph.flist train_trans.txt train_sph.scp || exit 1;
mv train_trans.txt tmp; sort -k 1 tmp > train_trans.txt
mv train_sph.scp tmp; sort -k 1 tmp > train_sph.scp
rm tmp

sph2pipe=`cd $S3_ROOT ; cd ../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -f $sph2pipe ]; then
    echo "Could not find the sph2pipe program at $sph2pipe";
    exit 1;
fi
awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < train_sph.scp > train_wav.scp

cat train_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > train.utt2spk
cat train.utt2spk | sort -k 2 | $S3_ROOT/scripts/utt2spk_to_spk2utt.pl > train.spk2utt

echo "Creating coretest set."
test_speakers="mdab0 mwbt0 felc0 mtas1 mwew0 fpas0 mjmp0 mlnt0 fpkt0 mlll0 mtls0 fjlm0 mbpm0 mklt0 fnlp0 mcmj0 mjdh0 fmgd0 mgrt0 mnjm0 fdhc0 mjln0 mpam0 fmld0"
dev_speakers="faks0 fdac1 fjem0 mgwt0 mjar0 mmdb1 mmdm2 mpdf0 fcmh0 fkms0 mbdg0 mbwm0 mcsh0 fadg0"
dev_speakers="${dev_speakers} fdms0 fedw0 mgjf0 mglb0 mrtk0 mtaa0 mtdt0 mthc0 mwjg0 fnmr0 frew0 fsem0 mbns0 mmjr0 mdls0 mdlf0"
dev_speakers="${dev_speakers} mdvc0 mers0 fmah0 fdrw0 mrcs0 mrjm4 fcal1 mmwh0 fjsj0 majc0 mjsw0 mreb0 fgjd0 fjmg0 mroa0 mteb0 mjfc0 mrjr0 fmml0 mrws1"


if [ $upper_case == 1 ] ; then
   test_speakers=`echo $test_speakers | tr '[:lower:]' '[:upper:]'`
   dev_speakers=`echo $dev_speakers | tr '[:lower:]' '[:upper:]'`
fi

rm -f test_sph.flist
for speaker in $test_speakers ; do
echo -n $speaker " "
(
   find $test_folder/*/${speaker} -iname "*.wav" | perl -ane 'if (! m/sa[0-9].wav/i){ print $_ ; }'
)  >> test_sph.flist
done 
echo ""
num_lines=`wc -l test_sph.flist | awk '{print $1}'`
echo "# of utterances in coretest set = ${num_lines}"

echo "Creating dev set."
rm -f dev_sph.flist
for speaker in $dev_speakers ; do
echo -n $speaker " "
(
   find $test_folder/*/${speaker} -iname "*.wav" | perl -ane 'if (! m/sa[0-9].wav/i){ print $_ ; }'
)  >> dev_sph.flist
done 
echo ""
num_lines=`wc -l dev_sph.flist | awk '{print $1}'`
echo "# of utterances in dev set = ${num_lines}"


# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
for test in test dev ; do
    echo "Finalizing ${test}"
    $S3_ROOT/local/make_trans.pl ${test} ${test}_sph.flist ${test}_trans.txt ${test}_sph.scp || exit 1;
    mv ${test}_trans.txt tmp; sort -k 1 tmp > ${test}_trans.txt
    mv ${test}_sph.scp tmp; sort -k 1 tmp > ${test}_sph.scp
    rm tmp;
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${test}_sph.scp  > ${test}_wav.scp

    cat ${test}_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > ${test}.utt2spk
    cat ${test}.utt2spk | sort -k 2 | $S3_ROOT/scripts/utt2spk_to_spk2utt.pl > ${test}.spk2utt
done


# Need to set these on the basis of file name first characters.
#grep -v "^;" DOC/SPKRINFO.TXT | awk '{print $1 " " $2 ; } ' | \
cat $spkr_info_file | \
    perl -ane 'tr/A-Z/a-z/;print;' | grep -v ';' | \
    awk '{print $2$1, $2}' | sort | uniq > spk2gender.map || exit 1;


echo timit_data_prep succeeded.
