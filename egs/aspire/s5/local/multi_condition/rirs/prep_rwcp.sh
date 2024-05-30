#!/usr/bin/env bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the RWCP impulse responses and ambient noise
# (Provided by MERL)
#-----------------------------------------
# all data is headerless binary type with little endian and 48 KHz
# impulse responses are float32 (4 byte)
# noises are short (2 byte)
# Data is multi-channel and each directory has a recording with each channel
# as a seperate file
# Impulse responses


download=true
sampling_rate=8k
output_bit=16
DBname=RWCP
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
  # RWCP sound scene database
  #==========================
  (cd $RIR_home;
  rm -rf RWCP.tar.gz
  wget http://www.openslr.org/resources/13/RWCP.tar.gz || exit 1;
  tar -zxvf RWCP.tar.gz >/dev/null
  )
fi

RWCP_home=$RIR_home/RWCP
RWCP_dirs=[]
RWCP_dirs[0]=$RWCP_home/micarray/MICARRAY/data1/ 
RWCP_dirs[1]=$RWCP_home/micarray/MICARRAY/data3/
RWCP_dirs[2]=$RWCP_home/micarray/MICARRAY/data5/


command_file=$log_dir/RWCP_read_rir_noise.sh
echo "">$command_file
# micarray database
type_num=0
for base_dir_name in ${RWCP_dirs[@]}; do
  type_num=$((type_num + 1))
  leaf_directories=( $(find $base_dir_name -type d -links 2 -print || exit -1) )
  files_done=0
  total_files=$(echo ${leaf_directories[@]}|wc -w)
  echo "Found ${total_files} impulse responses in ${base_dir_name}."
  echo "" > $log_dir/RWCP_type$type_num.rir.list
  # create the list of commands to be executed
  for leaf_dir_name in  ${leaf_directories[@]}; do
    first_channel=$(ls $leaf_dir_name|sed -e"s/.*\.//g"|sort -n|head -1)
    last_channel=$(ls $leaf_dir_name|sed -e"s/.*\.//g"|sort -nr|head -1)
    file_base_name=$(basename $leaf_dir_name)
    output_file_name=`echo ${leaf_dir_name#$base_dir_name}| sed -e"s/[\/\]\+/_/g" | tr '[:upper:]' '[:lower:]'`    
    output_file_name=RWCP_type${type_num}_rir_${output_file_name}.wav
    channel_files=
    for i in `seq $first_channel $last_channel`; do
      channel_files="$channel_files -t raw -e float -b 32 -c 1 -r 48k  $leaf_dir_name/$file_base_name.$i ";
    done
    echo "sox -M $channel_files -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
    echo ${output_dir}/${output_file_name} >>  $log_dir/RWCP_type$type_num.rir.list
    files_done=$((files_done + 1))
  done
done

# robot - non-directional microphone
# sox is not able to handle input scaling and there is lot of clipping
# so we scale the values in python first
tempdir_robo=`mktemp -d $PWD/tempXXXX`

cat << EOF > $tempdir_robo/raw_read.py 
import sys, numpy as np, argparse, scipy.signal as signal, os.path, glob, scipy.io, scipy.io.wavfile
precision = np.float32
file_handle = open(sys.argv[1], 'rb')
data = np.fromfile(file_handle, dtype = precision)
data = (0.9 * data / np.max(np.abs(data))) * (2**31)
data = data.astype('int32', copy = False)
scipy.io.wavfile.write(sys.argv[2], 48000, data) 
EOF

type_num=$((type_num + 1))
data_files=( $(find $RWCP_home/robot/data/non -name '*.dat' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/RWCP_type$type_num.rir.list
echo "Found $total_files impulse responses in ${RWCP_home}/robot/data/non."
# create the list of commands to be executed
for data_file in ${data_files[@]}; do
  temp_file=$tempdir_robo/$files_done.wav
  python $tempdir_robo/raw_read.py $data_file $temp_file 
  output_file_name=RWCP_type${type_num}_rir_`basename $data_file .dat | tr '[:upper:]' '[:lower:]'`.wav
  echo "sox -t wav $temp_file -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}"   >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/RWCP_type$type_num.rir.list
  files_done=$((files_done + 1))
done

# Ambient noise
type_num=$((type_num + 1))
base_dir_name=$RWCP_home/micarray/MICARRAY/data6/
leaf_directories=( $(find $base_dir_name -type d -links 2 -print || exit -1) )
files_done=0
total_files=$(echo ${leaf_directories[@]}|wc -w)
echo "" > $log_dir/RWCP_type$type_num.noise.list
echo "Found $total_files noises in ${base_dir_name}."
for leaf_dir_name in  ${leaf_directories[@]}; do
  first_channel=$(ls $leaf_dir_name|sed -e"s/.*\.//g"|sort -n|head -1)
  last_channel=$(ls $leaf_dir_name|sed -e"s/.*\.//g"|sort -nr|head -1)
  file_base_name=$(basename $leaf_dir_name)
  output_file_name=`echo ${leaf_dir_name#$base_dir_name}| sed -e"s/[\/\]\+/_/g" | tr '[:upper:]' '[:lower:]'`
  output_file_name=RWCP_type${type_num}_noise_${output_file_name}.wav
  channel_files=
  for i in `seq $first_channel $last_channel`; do
    channel_files="$channel_files -t raw -e signed-integer -b 16 -c 1 -r 48k  $leaf_dir_name/$file_base_name.$i ";
  done
  echo "sox -M $channel_files -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file

  echo ${output_dir}/${output_file_name} >>  $log_dir/RWCP_type$type_num.noise.list
  files_done=$((files_done + 1))
done

if [ ! -z "$file_splitter" ]; then
  num_jobs=$($file_splitter $command_file || exit 1)
  job_file=${command_file%.sh}.JOB.sh
else
  num_jobs=1
  job_file=$command_file
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

# get the RWCP database noise mic and room settings to pair with corresponding impulse responses
type_num=5
noise_patterns=( $(ls ${output_dir}/RWCP_type${type_num}_noise*.wav | xargs -n1 basename | python -c"
import sys
for line in sys.stdin:
  name = line.split('RWCP_type${type_num}_noise')[1]
  print '_'.join(name.split('_')[1:-1])
"|sort -u) )

for noise_pattern in ${noise_patterns[@]}; do
  set_file=$output_dir/info/noise_impulse_RWCP_$noise_pattern
  echo -n "noise_files=" > $set_file
  ls ${output_dir}/*${noise_pattern}*.wav | grep "type${type_num}" | grep "noise" | awk '{ ORS="  "; print; } END{print "\n"}' >> $set_file
  echo -n "impulse_files=" >> $set_file
  ls ${output_dir}/*${noise_pattern}*.wav | grep -v "type${type_num}" | grep "rir" | awk '{ ORS="  "; print; } END{print "\n"}' >> $set_file
done


# remove the tempdir we created to tackle the scaling problem
rm -rf $tempdir_robo
