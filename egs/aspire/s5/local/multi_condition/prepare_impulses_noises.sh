#!/bin/bash
# set -e

# Copyright 2014  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0.
# This script processes RIRs available from available databases into 
# 8Khz wav files so that these can be used to corrupt the Fisher data.
# The databases used are:
# RWCP :  http://research.nii.ac.jp/src/en/RWCP-SSD.html (this data is provided with the recipe. Thanks to Mitsubishi Electric Research Laboratories)
# AIRD : Aachen Impulse response database (http://www.ind.rwth-aachen.de/en/research/tools-downloads/aachen-impulse-response-database/)
# Reverb2014 : http://reverb2014.dereverberation.com/download.html
 
stage=0

log_dir=log
. utils/parse_options.sh


if [ $# -ne 1 ]; then
  echo "$0 outputdir"
  exit 1;
fi

output_dir=$1
mkdir -p $output_dir
mkdir -p $output_dir/info
mkdir -p $log_dir
rm -f $log_dir/type*.list

# download the necessary databases
if [ $stage -le 0 ]; then
  mkdir -p db/RIR_databases
  (cd db/RIR_databases
    # download the AIR database
    wget https://www2.ind.rwth-aachen.de/air/air_database_release_1_4.zip || exit 1;
    unzip air_database_release_1_4.zip > /dev/null
    # download reverb2014 database
    wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz || exit 1;
    tar -zxvf reverb_tools_for_Generate_mcTrainData.tgz
    wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_SimData.tgz || exit 1;
    tar -zxvf reverb_tools_for_Generate_SimData.tgz >/dev/null
    # download RWCP sound scene database
    wget http://www.openslr.org/resources/13/RWCP.tar.gz || exit 1;
    tar -zxvf RWCP.tar.gz >/dev/null
  )
fi

RWCP_home=db/RIR_databases/RWCP
AIR_home=db/RIR_databases/AIR_1_4
Reverb2014_home1=db/RIR_databases/reverb_tools_for_Generate_mcTrainData
Reverb2014_home2=db/RIR_databases/reverb_tools_for_Generate_SimData
# RWCP impulse responses and ambient noise
#-----------------------------------------
# all data is headerless binary type with little endian and 48 KHz
# impulse responses are float32 (4 byte)
# noises are short (2 byte)
# Data is multi-channel and each directory has a recording with each channel
# as a seperate file
sampling_rate=8000
# Impulse responses
RWCP_dirs=[]
RWCP_dirs[0]=$RWCP_home/micarray/MICARRAY/data1/ 
RWCP_dirs[1]=$RWCP_home/micarray/MICARRAY/data3/
RWCP_dirs[2]=$RWCP_home/micarray/MICARRAY/data5/


command_file=$log_dir/read_rir_noise.sh
echo "">$command_file
# micarray database
type_num=0
for base_dir_name in ${RWCP_dirs[@]}; do
  type_num=$((type_num + 1))
  leaf_directories=( $(find $base_dir_name -type d -links 2 -print || exit -1) )
  files_done=0
  total_files=$(echo ${leaf_directories[@]}|wc -w)
  echo "Found ${total_files} impulse responses in ${base_dir_name}."
  echo "" > $log_dir/type$type_num.rir.list
  # create the list of commands to be executed
  for leaf_dir_name in  ${leaf_directories[@]}; do
    output_file_name=`echo ${leaf_dir_name#$base_dir_name}| sed -e"s/[\/\]\+/_/g" | tr '[:upper:]' '[:lower:]'`
    output_file_name=type${type_num}_rir_${output_file_name}.wav
    echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate rwcp/rir ${leaf_dir_name} ${output_dir}/${output_file_name} || exit -1" >> $command_file 
    echo ${output_dir}/${output_file_name} >>  $log_dir/type$type_num.rir.list
    files_done=$((files_done + 1))
  done
done

# robot - non-directional microphone
type_num=$((type_num + 1))
data_files=( $(find $RWCP_home/robot/data/non -name '*.dat' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/type$type_num.rir.list
echo "Found $total_files impulse responses in ${RWCP_home}/robot/data/non."
# create the list of commands to be executed
for data_file in ${data_files[@]}; do
  output_file_name=type${type_num}_rir_`basename $data_file .dat | tr '[:upper:]' '[:lower:]'`.wav
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate rwcp/rir ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type$type_num.rir.list
  files_done=$((files_done + 1))
done

# Ambient noise
type_num=$((type_num + 1))
base_dir_name=$RWCP_home/micarray/MICARRAY/data6/
leaf_directories=( $(find $base_dir_name -type d -links 2 -print || exit -1) )
files_done=0
total_files=$(echo ${leaf_directories[@]}|wc -w)
echo "" > $log_dir/type$type_num.noise.list
echo "Found $total_files noises in ${base_dir_name}."
for leaf_dir_name in  ${leaf_directories[@]}; do
  output_file_name=`echo ${leaf_dir_name#$base_dir_name}| sed -e"s/[\/\]\+/_/g" | tr '[:upper:]' '[:lower:]'`
  output_file_name=type${type_num}_noise_${output_file_name}.wav
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate rwcp/noise ${leaf_dir_name} ${output_dir}/${output_file_name} || exit -1" >> $command_file;
  echo ${output_dir}/${output_file_name} >>  $log_dir/type$type_num.noise.list
  files_done=$((files_done + 1))
done

# AIR impulse responses
#----------------------
# data is stored in mat files with info about the impulse stored in a structure
type_num=$((type_num + 1))
python local/multi_condition/get_air_file_patterns.py $AIR_home > $log_dir/air_file_pattern
files_done=0
total_files=$(cat $log_dir/air_file_pattern|wc -l)
echo "" > $log_dir/type$type_num.rir.list
echo "Found $total_files impulse responses in ${AIR_home}."
while read file_pattern output_file_name; do
  output_file_name=`echo type${type_num}_$output_file_name| tr '[:upper:]' '[:lower:]'`
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate air '${file_pattern}' ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type$type_num.rir.list
  files_done=$((files_done + 1))
done < $log_dir/air_file_pattern

# Reverb2014 RIRs and noise
#--------------------------
# data is stored as multi-channel wav-files

# Simdata for training
#--------------------
type_num=$((type_num + 1))
data_files=( $(find $Reverb2014_home1/RIR -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/type${type_num}.rir.list
echo "Found $total_files impulse responses in ${Reverb2014_home1}/RIR."
for data_file in ${data_files[@]}; do
  output_file_name=type${type_num}_`basename $data_file | tr '[:upper:]' '[:lower:]'` 
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate reverb2014 ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type${type_num}.rir.list
  files_done=$((files_done + 1))
done

data_files=( $(find $Reverb2014_home1/NOISE -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/type${type_num}.noise.list
echo "Found $total_files noises in ${Reverb2014_home1}/NOISE."
for data_file in ${data_files[@]}; do
  output_file_name=type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate reverb2014 ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type${type_num}.noise.list
  files_done=$((files_done + 1))
done

# Simdata for devset
type_num=$((type_num + 1))
data_files=( $(find $Reverb2014_home2/RIR -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/type${type_num}.rir.list
echo "Found $total_files impulse responses in ${Reverb2014_home2}/RIR."
for data_file in ${data_files[@]}; do
  output_file_name=type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate reverb2014 ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type${type_num}.rir.list
  files_done=$((files_done + 1))
done


data_files=( $(find $Reverb2014_home2/NOISE -name '*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/type${type_num}.noise.list
echo "Found $total_files noises in ${Reverb2014_home2}/NOISE."
for data_file in ${data_files[@]}; do
  output_file_name=type${type_num}_`basename $data_file | tr '[:upper:]' '[:lower:]'`
  echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate reverb2014 ${data_file} ${output_dir}/${output_file_name} || exit -1;" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/type${type_num}.noise.list
  files_done=$((files_done + 1))
done


# Running commands created above
echo "Extracting the found room impulse responses and noises."
# write the file_splitter to create jobs files for queue.pl
cat << EOF > $log_dir/file_splitter.py
import os.path, sys, math

input_file = sys.argv[1]
num_lines_per_file = int(sys.argv[2])
[file_base_name, ext] = os.path.splitext(input_file)
lines = open(input_file).readlines();
num_lines = len(lines)
num_jobs = int(math.ceil(num_lines/ float(num_lines_per_file)))

# filtering commands into seperate task files
for i in xrange(1, num_jobs+1) :
  cur_lines = map(lambda index: lines[index], range(i - 1, num_lines , num_jobs))
  file = open("{0}.{1}{2}".format(file_base_name, i, ext), 'w')
  file.write("source /home/vpeddinti/scripts//python/MyPython/bin/activate\n")
  file.write("which python\n")
  file.write("".join(cur_lines))
  file.close()
print num_jobs
EOF



num_jobs=$(python $log_dir/file_splitter.py $command_file 10 || exit 1)
# execute the commands using the above created array jobs
time $decode_cmd --max-jobs-run 40 JOB=1:$num_jobs $log_dir/log/read_rir_noise.JOB.log \
  sh $log_dir/read_rir_noise.JOB.sh || exit 1;
sleep 4;
echo "Normalizing the extracted room impulse responses and noises, per type"
for i in `ls $log_dir/type*.rir.list`; do
  echo "$i"
  python local/multi_condition/normalize_wavs.py --is-room-impulse-response true $i
done
 
for i in `ls $log_dir/type*.noise.list`; do
  echo "$i"
  python local/multi_condition/normalize_wavs.py --is-room-impulse-response false $i
done

cat $log_dir/type{1,2,3,4,6,7,8}.rir.list > $output_dir/info/impulse_files
cat $log_dir/type{5,7,8}.noise.list > $output_dir/info/noise_files


# get the Reverb2014 room names to pair the noises and impulse responses 
for type_num in `seq 7 8`; do
  noise_patterns=( $(ls ${output_dir}/type${type_num}_noise*.wav | xargs -n1 basename | python -c"
import sys
for line in sys.stdin:
  name = line.split('type${type_num}_noise_')[1]
  print name.split('_')[0]
  "|sort -u) )
  for noise_pattern in ${noise_patterns[@]}; do
    set_file=$output_dir/info/noise_impulse_$noise_pattern
    echo -n "noise_files =" > ${set_file}
    ls ${output_dir}/type${type_num}_noise*${noise_pattern}*.wav | awk '{ ORS="  "; print;} END{print "\n"}' >> ${set_file}
    echo -n "impulse_files =" >> ${set_file}
    ls ${output_dir}/type${type_num}_rir*${noise_pattern}*.wav | awk '{ ORS="  "; print; } END{print "\n"}' >> ${set_file}
  done
done

# get the RWCP database noise mic and room settings to pair with corresponding impulse responses
type_num=5
noise_patterns=( $(ls ${output_dir}/type${type_num}_noise*.wav | xargs -n1 basename | python -c"
import sys
for line in sys.stdin:
  name = line.split('type${type_num}_noise')[1]
  print '_'.join(name.split('_')[1:-1])
"|sort -u) )

for noise_pattern in ${noise_patterns[@]}; do
  set_file=$output_dir/info/noise_impulse_$noise_pattern
  echo -n "noise_files=" > $set_file
  ls ${output_dir}/*${noise_pattern}*.wav | grep "type${type_num}" | grep "noise" | awk '{ ORS="  "; print; } END{print "\n"}' >> $set_file
  echo -n "impulse_files=" >> $set_file
  ls ${output_dir}/*${noise_pattern}*.wav | grep -v "type${type_num}" | grep "rir" | awk '{ ORS="  "; print; } END{print "\n"}' >> $set_file
done
