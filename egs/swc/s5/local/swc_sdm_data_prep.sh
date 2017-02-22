#!/bin/bash 

# Copyright 2016, The University of Sheffield (Author: Yulan Liu)
# SWC SDM data preparation (train, dev and eval), as well as scoring pareparation.
# By default, channel TBL1-01 is used.
# Apache 2.0

# To be run from one directory above this script.

. path.sh

export LC_ALL=C

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: swc_ihm_data_prep.sh  <path to SWC data folder>  <mode>"
  exit 1;
fi

SWC_DIR=$1	# SWC1, SWC2, SWC3 should be in subfolders under this directory
MODE=$2		

dir=data/sdm/$MODE/	# Output dir
mkdir -p $dir


# ---------------------------------------------------------------------------
# Explanation of the "MODE":
# 
#	SA: stand-alone
#	AD: adaptation
#
# SWC1, SWC2 and SWC3 all have three strips: A, B, C. 
#
# 	MODE		Train		Dev		Eval
#
# 	SA1		SWC1		
#			SWC2.A		SWC2.B		SWC2.C
#			SWC3.A		SWC3.B		SWC3.C
#	
#	SA2		SWC1
#					SWC2.A		SWC2.B+C
#					SWC3.A		SWC3.B+C
#
#	AD1		-		SWC1.A+B	SWC1.C
#					SWC2.A+B	SWC2.C
#					SWC3.A+B	SWC3.C
#
#	AD2		-		SWC1		
#							SWC2
#							SWC3
#
# ---------------------------------------------------------------------------


if [ ! -d $SWC_DIR ]; then
  echo "Error: cannot find SWC data foder: $SWC_DIR"
  exit 1;
fi


# Audio data directory check
for i in {1..3}
do
  if [ ! -d ${SWC_DIR}/swc${i}/audio ]; then
    echo "Error: cannot find audio folder for SWC${i}: ${SWC_DIR}/swc${i}/audio/"
    exit 1;
  fi
done


# And transcripts check
for i in {1..3}
do
  if [ ! -e ${SWC_DIR}/swc${i}/transcripts/swc${i}.stm ]; then
    echo "Error: cannot find transcripts for SWC${i}: ${SWC_DIR}/swc${i}/transcripts/swc${i}.stm"
    exit 1;
  fi
done


# find headset wav audio files only, here we again get all
# the files in the corpora and filter only specific sessions
# while building segments
CH='TBL1-01'		
echo "...Default channel for SDM: $CH"
ls ${SWC_DIR}/swc?/audio/table/*${CH}.wav | sort -u > $dir/wav.flist
n=`cat $dir/wav.flist | wc -l`
echo "In total, $n headset files were found."
[ $n -ne 24 ] && \
  echo "Warning: expected 24 (10 ses x 1 mics + 8 ses x 1 mics + 6 ses x 1 mics) data files, found $n"


echo "Preparing data for experiments in the $MODE mode."
if [ "$MODE" == "SA1" ] ; then
  # --- train: swc1.A+B+C, swc2.A, swc3.A 
  outdir=$dir/train
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir 
  grep -v ';;' ${SWC_DIR}/swc1/transcripts/swc1_speech.stm > tmp
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,A>' >> tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,A>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

  # --- dev: swc2.B, swc3.B
  outdir=$dir/dev
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,B>' > tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,B>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm


  # --- eval: swc2.C, swc3.C
  outdir=$dir/eval
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,C>' > tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,C>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm


elif [ "$MODE" == "SA2" ] ; then
  # --- train: swc1.A+B+C
  outdir=$dir/train
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc1/transcripts/swc1_speech.stm > tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

  # --- dev: swc2.A, swc3.A
  outdir=$dir/dev
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,A>' > tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,A>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

  # --- eval: swc2.B+C, swc3.B+C
  outdir=$dir/eval
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,B>\|<swc2,C>' > tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,B>\|<swc3,C>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm


elif [ "$MODE" == "AD1" ] ;  then
  # --- dev: swc1.A+B, swc2.A+B, swc3.A+B
  outdir=$dir/dev
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc1/transcripts/swc1_speech.stm | grep '<swc1,A>\|<swc1,B>' > tmp
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,A>\|<swc2,B>' >> tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,A>\|<swc3,B>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

  # --- eval: swc1.C, swc2.C, swc3.C
  outdir=$dir/eval
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc1/transcripts/swc1_speech.stm | grep '<swc1,C>' > tmp
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm | grep '<swc2,C>' >> tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm | grep '<swc3,C>' >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

elif [ "$MODE" == "AD2" ] ; then
  # --- dev: swc1.A+B+C
  outdir=$dir/dev
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc1/transcripts/swc1_speech.stm > tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp > $outdir/stm

  # --- eval: swc2.A+B+C, swc3.A+B+C
  outdir=$dir/eval
  echo -e "\nprepare folder $outdir"
  mkdir -p $outdir
  grep -v ';;' ${SWC_DIR}/swc2/transcripts/swc2_speech.stm  > tmp
  grep -v ';;' ${SWC_DIR}/swc3/transcripts/swc3_speech.stm  >> tmp
  awk '{if ($1==";;"){print $0}else{$2="A"; print $0}}' tmp | grep -v mn0010 > $outdir/stm

else
  echo "Error: unrecognized mode: $MODE (you can only use \"SA1\", \"SA2\", \"AD1\" or \"AD2\" with this script)"
  exit 1;
fi

for i in 'train' 'dev' 'eval'
do
  outdir=$dir/$i
  if [ -d $outdir ] ; then

# SWC1-00001 01 mn0001 1.10 1.80 <swc1,A> YOU'RE NUMBER TWO
  awk '{printf("%s_%s_%06d_%06d", $3, $1, int($4*100+0.5), int($5*100+0.5)); for(i=7;i<=NF; i++){printf(" %s", $i);} printf("\n")}' $outdir/stm | sed 's/(//g' |sed 's/)//g' | sort -u  > $outdir/text
  awk '{printf("%s_%s_%06d_%06d %s_TBL1-01 %.2f %.2f\n", $3, $1, int($4*100+0.5), int($5*100+0.5), $1, $4, $5)}' $outdir/stm | sort -u  > $outdir/segments
  awk '{printf("%s_%s_%06d_%06d %s\n", $3, $1, int($4*100+0.5), int($5*100+0.5), $3)}' $outdir/stm | sort -u  > $outdir/utt2spk
  sort -k 2 $outdir/utt2spk | utils/utt2spk_to_spk2utt.pl > $outdir/spk2utt || exit 1;

  awk '{print $2}' $outdir/segments | sort -u > tmp
  grep 'SWC1' tmp | awk -v fd=${SWC_DIR}/swc1/audio/table '{print $0" sox -c 1 -t wavpcm -s "fd"/"$0".wav -t wavpcm - | "}' > tmp2
  grep 'SWC2' tmp | awk -v fd=${SWC_DIR}/swc2/audio/table '{print $0" sox -c 1 -t wavpcm -s "fd"/"$0".wav -t wavpcm - | "}' >> tmp2
  grep 'SWC3' tmp | awk -v fd=${SWC_DIR}/swc3/audio/table '{print $0" sox -c 1 -t wavpcm -s "fd"/"$0".wav -t wavpcm - | "}' >> tmp2

  sort -u tmp2 > $outdir/wav.scp
  rm tmp*
 
  utils/validate_data_dir.sh --no-feats $outdir || exit 1;


  # Files needed for scoring
  cp local/en20140220.glm  $outdir/glm
  # segments: fn0016_SWC3-00001_181870_181927 SWC3-00001_fn0016 1818.70 1819.27
# ------- This is the first version I thought should be, but later it does not pass valicaiton script ("utils/validate_data_dir.sh")
#  # reco2file_and_chan: SWC3-00001_fn0016  SWC3-00001  0                # NOTE: IHM is different here! 0=>16
#  awk '{print $2}' $outdir/segments | sort -u | awk 'BEGIN{FS="_"}{print $0" "$1" 0"}' > $outdir/reco2file_and_channel
#
# ------- This is the updated hacked version 
  # reco2file_and_chan: SWC3-00001_fn0016  SWC3-00001  A              # NOTE: IHM is different here! Second column => SWC3-00001_fn0016
  awk '{print $2}' $outdir/segments | sort -u | awk 'BEGIN{FS="_"}{print $0" "$1" A"}' > $outdir/reco2file_and_channel

  echo ";;
;;
;; CATEGORY \"0\" \"$MODE ($i set)\" \"\"
;; LABEL \"O\" \"Overall\" \"Overall\"
;;
;;
;; CATEGORY \"1\" \"Strip\" \"\"
;; LABEL \"A\" \"A\" \"\"
;; LABEL \"B\" \"B\" \"\"
;; LABEL \"C\" \"C\" \"\"
;;
;;
;; CATEGORY \"2\" \"Release\" \"\"
;; LABEL \"swc1\" \"swc1\" \"\"
;; LABEL \"swc2\" \"swc2\" \"\"
;; LABEL \"swc3\" \"swc3\" \"\" " > tmp_stm


# -------- This is the first version I thought should be, but the reco2file_and_channel file for this versiont does not pass valicaiton script ("utils/validate_data_dir.sh")
#  mv $outdir/stm tmp
# ------- This is the updated hacked version 
  awk '{if($1==";;"){print $0}else{$2="A"; print $0}}' $outdir/stm > tmp
  cat tmp_stm tmp > $outdir/stm


  fi
done

rm tmp*

echo "SWC SDM data preparation succeeded."

exit 0;



