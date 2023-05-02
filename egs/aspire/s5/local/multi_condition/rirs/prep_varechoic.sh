#!/usr/bin/env bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the impulse responses from the Varechoic room
# available at
# http://www1.icsi.berkeley.edu/Speech/papers/gelbart-ms/pointers/

download=true
sampling_rate=8k
output_bit=16
DBname=VARECHOIC
file_splitter=  #script to generate job scripts given the command file

. ./cmd.sh
. ./path.sh
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

if [ "$download" = true ]; then
  mkdir -p $RIR_home
  (cd $RIR_home;

  dir=icsi_varechoic/
  rm -rf $dir
  wget http://www.icsi.berkeley.edu/ftp/global/pub/speech/papers/gelbart-ms/pointers/varechoic.zip --directory-prefix=$dir
  (cd $dir;
    unzip varechoic.zip
  )
  )
fi

command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file
type_num=1
echo "" > $log_dir/${DBname}_type$type_num.rir.list
varechoic_home=$RIR_home/icsi_varechoic/varechoic
for room_type in ir00 ir43 ir100 ; do
  for mike in m1 m2 m3 m4; do
    file_basename=${room_type}${mike}
    echo "sox  -B -e float -b 32 -c 1 -r 8k -t raw $varechoic_home/${file_basename}.raw -t wav -b $output_bit $output_dir/${DBname}_${file_basename}.wav" >> $command_file
    echo $output_dir/${DBname}_${file_basename}.wav >>  $log_dir/${DBname}_type$type_num.rir.list
  done
done

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
