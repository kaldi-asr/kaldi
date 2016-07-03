#!/bin/bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads Multichannel Acoustic Reverberation Database at
# http://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/
#==============================================

download=true
sampling_rate=8k
output_bit=16
DBname=MARDY
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

if [ "$download" = true ]; then
  mkdir -p $RIR_home
  (cd $RIR_home;
  dir=mardy/
  rm -rf $dir
  wget http://www.commsp.ee.ic.ac.uk/~sap/uploads/data/MARDY.rar --directory-prefix=$dir

  cd $dir
  unrar e MARDY.rar
  )
fi

command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file
type_num=1
data_files=( $(find $RIR_home/mardy/ -name '*.wav' -type f -print || exit -1) )
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files impulse responses in ${RIR_home}/mardy/"
file_count=1
for data_file in ${data_files[@]}; do
  #output_file_name=${DBname}_type${type_num}_${file_count}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  output_file_name=${DBname}_type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  #echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate wav ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.rir.list
  file_count=$((file_count + 1))
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
