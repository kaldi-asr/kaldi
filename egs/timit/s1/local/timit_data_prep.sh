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

mkdir -p data/local
cd data/local


if [ ! -d $TIMIT_ROOT/TIMIT/TRAIN -o ! -d $TIMIT_ROOT/TIMIT/TEST ]; then
   echo "Error: run.sh requires a directory argument (an absolute pathname) that contains TIMIT/TRAIN and TIMIT/TEST"
   exit 1; 
fi  

(
   find $TIMIT_ROOT/TIMIT/TRAIN -name "*.WAV" | perl -ane 'if (! m/SA[0-9].WAV/){ print $_ ; }'
)  > train_sph.flist


# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
../../local/make_trans.pl trn train_sph.flist train_trans.txt train_sph.scp
mv train_trans.txt tmp; sort -k 1 tmp > train_trans.txt
mv train_sph.scp tmp; sort -k 1 tmp > train_sph.scp
rm tmp

sph2pipe=`cd ../../../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -f $sph2pipe ]; then
    echo "Could not find the sph2pipe program at $sph2pipe";
    exit 1;
fi
awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < train_sph.scp > train_wav.scp

cat train_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > train.utt2spk
cat train.utt2spk | sort -k 2 | ../../scripts/utt2spk_to_spk2utt.pl > train.spk2utt

echo "Creating coretest set."
rm -f test_sph.flist
test_speakers="MDAB0 MWBT0 FELC0 MTAS1 MWEW0 FPAS0 MJMP0 MLNT0 FPKT0 MLLL0 MTLS0 FJLM0 MBPM0 MKLT0 FNLP0 MCMJ0 MJDH0 FMGD0 MGRT0 MNJM0 FDHC0 MJLN0 MPAM0 FMLD0"

for speaker in $test_speakers ; do
echo -n $speaker " "
(
    find $TIMIT_ROOT/TIMIT/TEST/*/${speaker}  -name "*.WAV" | perl -ane 'if (! m/SA[0-9].WAV/){ print $_ ; }'
)  >> test_sph.flist
done 
echo ""
num_lines=`wc -l test_sph.flist | awk '{print $1}'`
echo "# of utterances in coretest set = ${num_lines}"

echo "Creating dev set."
dev_speakers="FAKS0 FDAC1 FJEM0 MGWT0 MJAR0 MMDB1 MMDM2 MPDF0 FCMH0 FKMS0 MBDG0 MBWM0 MCSH0 FADG0"
dev_speakers="${dev_speakers} FDMS0 FEDW0 MGJF0 MGLB0 MRTK0 MTAA0 MTDT0 MTHC0 MWJG0 FNMR0 FREW0 FSEM0 MBNS0 MMJR0 MDLS0 MDLF0"
dev_speakers="${dev_speakers} MDVC0 MERS0 FMAH0 FDRW0 MRCS0 MRJM4 FCAL1 MMWH0 FJSJ0 MAJC0 MJSW0 MREB0 FGJD0 FJMG0 MROA0 MTEB0 MJFC0 MRJR0 FMML0 MRWS1"

rm -f dev_sph.flist
for speaker in $dev_speakers ; do
echo -n $speaker " "
(
    find $TIMIT_ROOT/TIMIT/TEST/*/${speaker}  -name "*.WAV" | perl -ane 'if (! m/SA[0-9].WAV/){ print $_ ; }'
)  >> dev_sph.flist
done 
echo ""
num_lines=`wc -l dev_sph.flist | awk '{print $1}'`
echo "# of utterances in dev set = ${num_lines}"


# make_trans.pl also creates the utterance id's and the kaldi-format scp file.
for test in test dev ; do
    echo "Finalizing ${test}"
    ../../local/make_trans.pl ${test} ${test}_sph.flist ${test}_trans.txt ${test}_sph.scp
    mv ${test}_trans.txt tmp; sort -k 1 tmp > ${test}_trans.txt
    mv ${test}_sph.scp tmp; sort -k 1 tmp > ${test}_sph.scp
    rm tmp;
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${test}_sph.scp  > ${test}_wav.scp

    cat ${test}_wav.scp | perl -ane 'm/^(\w+_(\w+)\w_\w+) / || die; print "$1 $2\n"' > ${test}.utt2spk
    cat ${test}.utt2spk | sort -k 2 | ../../scripts/utt2spk_to_spk2utt.pl > ${test}.spk2utt
done


# Need to set these on the basis of file name first characters.
#grep -v "^;" DOC/SPKRINFO.TXT | awk '{print $1 " " $2 ; } ' | \
cat $TIMIT_ROOT/TIMIT/DOC/SPKRINFO.TXT | \
    perl -ane 'tr/A-Z/a-z/;print;' | grep -v ';' | \
    awk '{print $2$1, $2}' | sort | uniq > spk2gender.map || exit 1;

# NEED TO DO THE FOLLOWING TWO STEPS
# USE THE SWBD RECIPE FOR THIS. SEE local/swbd_p1_train_lms.sh file I COPIED
#../../scripts/make_rm_lm.pl $TIMIT_ROOT/rm1_audio1/rm1/doc/wp_gram.txt  > G.txt || exit 1;

# Getting lexicon
#../../scripts/make_rm_dict.pl  $TIMIT_ROOT/rm1_audio2/2_4_2/score/src/rdev/pcdsril.txt \
    #> lexicon.txt || exit 1;



echo timit_data_prep succeeded.
