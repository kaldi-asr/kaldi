#!/usr/bin/env bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the impulse responses and noise files from the
# Reverb2014 challenge
# and converts them to wav files with the required sampling rate
#==============================================

download=true
sampling_rate=8k
output_bit=16
DBname=RVB2014
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
  rm -rf reverb_tools*.tgz
  wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz || exit 1;
  tar -zxvf reverb_tools_for_Generate_mcTrainData.tgz
  wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_SimData.tgz || exit 1;
  tar -zxvf reverb_tools_for_Generate_SimData.tgz >/dev/null
  )
fi

Reverb2014_home1=$RIR_home/reverb_tools_for_Generate_mcTrainData
Reverb2014_home2=$RIR_home/reverb_tools_for_Generate_SimData

# Reverb2014 RIRs and noise
#--------------------------
# data is stored as multi-channel wav-files

command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file
# Simdata for training
#--------------------
type_num=1
data_files=( $(find $Reverb2014_home1/RIR -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files impulse responses in ${Reverb2014_home1}/RIR."
for data_file in ${data_files[@]}; do
  output_file_name=${DBname}_type${type_num}_`basename $data_file | tr '[:upper:]' '[:lower:]'` 
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.rir.list
  files_done=$((files_done + 1))
done

data_files=( $(find $Reverb2014_home1/NOISE -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.noise.list
echo "Found $total_files noises in ${Reverb2014_home1}/NOISE."
for data_file in ${data_files[@]}; do
  output_file_name=${DBname}_type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.noise.list
  files_done=$((files_done + 1))
done

# Simdata for devset
type_num=$((type_num + 1))
data_files=( $(find $Reverb2014_home2/RIR -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files impulse responses in ${Reverb2014_home2}/RIR."
for data_file in ${data_files[@]}; do
  output_file_name=${DBname}_type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.rir.list
  files_done=$((files_done + 1))
done


data_files=( $(find $Reverb2014_home2/NOISE -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.noise.list
echo "Found $total_files noises in ${Reverb2014_home2}/NOISE."
for data_file in ${data_files[@]}; do
  output_file_name=${DBname}_type${type_num}_`basename $data_file | tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.noise.list
  files_done=$((files_done + 1))
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

# get the Reverb2014 room names to pair the noises and impulse responses 
for type_num in `seq 1 2`; do
  noise_patterns=( $(ls ${output_dir}/${DBname}_type${type_num}_noise*.wav | xargs -n1 basename | python -c"
import sys
for line in sys.stdin:
  name = line.split('${DBname}_type${type_num}_noise_')[1]
  print name.split('_')[0]
  "|sort -u) )
  for noise_pattern in ${noise_patterns[@]}; do
    set_file=$output_dir/info/noise_impulse_${DBname}_$noise_pattern
    echo -n "noise_files =" > ${set_file}
    ls ${output_dir}/${DBname}_type${type_num}_noise*${noise_pattern}*.wav | awk '{ ORS="  "; print;} END{print "\n"}' >> ${set_file}
    echo -n "impulse_files =" >> ${set_file}
    ls ${output_dir}/${DBname}_type${type_num}_rir*${noise_pattern}*.wav | awk '{ ORS="  "; print; } END{print "\n"}' >> ${set_file}
  done
done

