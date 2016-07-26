#!/bin/bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the Aachen impulse response database
# http://www.ind.rwth-aachen.de/en/research/tools-downloads/aachen-impulse-response-database/
#==============================================


download=true
sampling_rate=8000
DBname=AIR
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
  rm -rf air_database_release_1_4.zip
  wget http://www.openslr.org/resources/20/air_database_release_1_4.zip || exit 1;
  unzip air_database_release_1_4.zip > /dev/null
  )
fi

AIR_home=$RIR_home/AIR_1_4
# AIR impulse responses
#----------------------
# data is stored in mat files with info about the impulse stored in a structure
type_num=1
python local/multi_condition/get_air_file_patterns.py $AIR_home > $log_dir/air_file_pattern
files_done=0
total_files=$(cat $log_dir/air_file_pattern|wc -l)
echo "" > $log_dir/${DBname}_type$type_num.rir.list
echo "Found $total_files impulse responses in ${AIR_home}."

command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file
file_count=1
while read file_pattern output_file_name; do
 # output_file_name=`echo ${DBname}_type${type_num}_${file_count}_$output_file_name| tr '[:upper:]' '[:lower:]'`
  output_file_name=`echo ${DBname}_type${type_num}_$output_file_name| tr '[:upper:]' '[:lower:]'`
  echo "local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate air '${file_pattern}' ${output_dir}/${output_file_name} || exit 1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type$type_num.rir.list
  file_count=$((file_count + 1))
done < $log_dir/air_file_pattern

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
