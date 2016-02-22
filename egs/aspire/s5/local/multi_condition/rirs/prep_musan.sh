#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0
# This script downloads the MUSAN music and noise database
#-----------------------------------------
# all data is wav at 16kHZ

download=true
sampling_rate=8k
output_bit=16
DBname=MUSAN
file_splitter=  #script to generate job scripts given the command file

. cmd.sh
. path.sh
. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <rir-home> <output-dir> <log-dir>"
  echo "e.g.:"
  echo " $0  --download true db/RIR_databases/ data/impulses_noises exp/make_reverb/log"
  exit 1;
fi

RIR_home=$1
output_dir=$2
log_dir=$3

mkdir -p $log_dir
mkdir -p $output_dir/info

if [ "$download" == true ]; then
  mkdir -p $RIR_home
  # MUSAN sound scene database
  #==========================
  (cd $RIR_home;
  rm -rf musan.tar.gz 
  wget http://www.openslr.org/resources/17/musan.tar.gz || exit 1;
  tar -zxvf musan.tar.gz >/dev/null
  )
fi

MUSAN_home=$RIR_home/musan

[ ! -d $MUSAN_home/noise ] && echo "$0: noise files not downloaded" && exit 1

find $MUSAN_home/noise -name "*.wav" -type f > $log_dir/${DBname}_noise.list
for x in `cat $log_dir/${DBname}_noise.list`; do
  y=`basename $x`
  z=${y%*.wav}
  echo "$z $x"
done | sort -k 1,1 > $log_dir/${DBname}_noise.scp

# Read the list of background noises
if [ `cat $MUSAN_home/noise/free-sound/ANNOTATIONS | wc -l` -ne 38 ]; then
  echo "$0: expected 38 lines in file; probably the corpus got updated and hence this script must be modified"
  exit 1
fi

cat $MUSAN_home/noise/free-sound/ANNOTATIONS | tail -n +2 | sort -u > $log_dir/${DBname}_background_noise.fileids

command_file=$log_dir/${DBname}_read_rir_noise.sh

echo -n "">$command_file

# Ambient noise
base_dir_name=$MUSAN_home/noise/free-sound/

echo -n "" > $output_dir/info/${DBname}.background.noise.list
background_noise_files_done=0
for x in `cat $log_dir/${DBname}_background_noise.fileids`; do 
  output_filename=$x.wav
  [ ! -f $base_dir_name/$x.wav ] && echo "$0: could not find file $x.wav in $base_dir_name" && exit 1
  echo "sox $base_dir_name/$x.wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_filename}" >> $command_file
  echo ${output_dir}/${output_filename} >> $output_dir/info/${DBname}.background.noise.list
  background_noise_files_done=$((background_noise_files_done + 1))
done

echo -n "" > $output_dir/info/${DBname}.foreground.noise.list
foreground_noise_files_done=0
utils/filter_scp.pl --exclude $log_dir/${DBname}_background_noise.fileids \
  $log_dir/${DBname}_noise.scp | \
while IFS=$'\n' read x; do
  file_id=`echo $x | awk '{print $1}'`
  file=`echo $x | awk '{print $2}'`
  output_filename=$file_id.wav

  [ ! -f $file ] && echo "$0: could not find file $file" && exit 1 
  echo "sox $file -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_filename}" >> $command_file
  echo ${output_dir}/${output_filename} >> $output_dir/info/${DBname}.foreground.noise.list
  foreground_noise_files_done=$((foreground_noise_files_done + 1))
done

background_noise_files_done=`cat $output_dir/info/${DBname}.background.noise.list | wc -l`
foreground_noise_files_done=`cat $output_dir/info/${DBname}.foreground.noise.list | wc -l`

echo "$0: read $foreground_noise_files_done foreground noise and $background_noise_files_done background noise files"

if [ "$foreground_noise_files_done" -eq 0 ] || [ $background_noise_files_done -eq 0 ]; then
  echo "$0: failed reading noise files from ${DBname} corpus"
  exit 1
fi

if [ ! -z "$file_splitter" ]; then
  num_jobs=$($file_splitter $command_file || exit 1)
  job_file=${command_file%.sh}.JOB.sh
  job_log=${command_file%.sh}.JOB.log
else
  num_jobs=1
  job_file=$command_file
  job_log=${command_file%.sh}.log
fi

# execute the commands using the above created array jobs
time $decode_cmd --max-jobs-run 40 JOB=1:$num_jobs $job_log \
  sh $job_file || exit 1;
