#!/bin/bash
# Copyrigh       2017  Ignacio Viñals
#           2017-2018  David Snyder
#
# This script prepares the SITW data.  It creates separate directories
# for dev enroll, eval enroll, dev test, and eval test.  It also prepares
# multiple trials files, in the test directories, but we usually only use the
# core-core.lst list.

if [  $# != 2 ]; then
    echo "Usage: make_sitw.sh <SITW_PATH> <this_out_dir>"
    echo "E.g.: make_sitw.sh /export/corpora/SRI/sitw data"
    exit 1
fi

in_dir=$1
out_dir=$2

# Prepare the enrollment data
for mode in dev eval; do
  this_out_dir=${out_dir}/sitw_${mode}_enroll
  mkdir -p $this_out_dir 2>/dev/null
  WAVFILE=$this_out_dir/wav.scp
  SPKFILE=$this_out_dir/utt2spk
  MODFILE=$this_out_dir/utt2cond
  rm $WAVFILE $SPKFILE $MODFILE 2>/dev/null
  this_in_dir=${in_dir}/$mode

  for enroll in core assist; do
    cat $this_in_dir/lists/enroll-${enroll}.lst | \
    while read line; do
      wav_id=`echo $line| awk '{print $2}' |\
        awk 'BEGIN{FS="[./]"}{print $(NF-1)}'`
      spkr_id=`echo $line| awk '{print $1}'`
      WAV=`echo $line | awk '{print this_in_dir"/"$2}' this_in_dir=$this_in_dir`
      echo "${spkr_id}_${wav_id} sox -t flac $WAV -t wav -r 16k -b 16 - channels 1 |" >> $WAVFILE
      echo "${spkr_id}_${wav_id} ${spkr_id}" >> $SPKFILE
      echo "${spkr_id}_${wav_id} $enroll $mode" >> $MODFILE
    done
  done
  utils/fix_data_dir.sh $this_out_dir
done

# Prepare the test data
for mode in dev eval; do
  this_out_dir=${out_dir}/sitw_${mode}_test
  mkdir -p $this_out_dir 2>/dev/null
  WAVFILE=$this_out_dir/wav.scp
  SPKFILE=$this_out_dir/utt2spk
  MODFILE=$this_out_dir/utt2cond
  rm $WAVFILE $SPKFILE $MODFILE 2>/dev/null
  mkdir -p $this_out_dir/trials 2>/dev/null
  mkdir -p $this_out_dir/trials/aux 2>/dev/null
  this_in_dir=${in_dir}/$mode

  for trial in core multi; do
    cat $this_in_dir/lists/test-${trial}.lst | awk '{print $1,$2}' |\
    while read line; do
      wav_id=`echo $line | awk 'BEGIN{FS="[./]"} {print $(NF-1)}'`
      WAV=`echo $line | awk '{print this_in_dir"/"$1}' this_in_dir=$this_in_dir`
      echo "${wav_id} sox -t flac $WAV -t wav -r 16k -b 16 - channels 1 |" >> $WAVFILE
      echo "${wav_id} ${wav_id}" >> $SPKFILE
      echo "${wav_id} $trial $mode" >> $MODFILE
    done
  done

  for trial in core-core core-multi assist-core assist-multi; do
    cat $this_in_dir/keys/$trial.lst | sed 's@audio/@@g' | sed 's@.flac@@g' |\
    awk '{if ($3=="tgt")
           {print $1,$2,"target"}
         else
           {print $1,$2,"nontarget"}
         }'   > $this_out_dir/trials/${trial}.lst
  done

  for trial in $this_in_dir/keys/aux/* ; do
    trial_name=`basename $trial`
    cat $trial | sed 's@audio/@@g' | sed 's@.flac@@g' |\
    awk '{if ($3=="tgt")
           {print $1,$2,"target"}
         else
           {print $1,$2,"nontarget"}
     }'   > $this_out_dir/trials/aux/${trial_name}
  done
  utils/fix_data_dir.sh $this_out_dir
done
