#!/usr/bin/env bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# Room impulse responses from Center for Digital Music, Queen Mary University of London
# Stewart, Rebecca and Sandler, Mark. "Database of Omnidirectional and B-Format Impulse Responses", in Proc. of IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2010), Dallas, Texas, March 2010.
# "These IRs are released under the Creative Commons Attribution-Noncommercial-Share-Alike license with attribution to the Centre for Digital Music, Queen Mary, University of London."
# http://isophonics.net/content/room-impulse-response-data-set

download=true 
sampling_rate=8k
output_bit=16
DBname=C4DM
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
  rm -rf c4dm
  #great hall
  #---------
  dir=c4dm/greathall/
  wget http://kakapo.dcs.qmul.ac.uk/irs/greathallOmni.zip  --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/greathallW.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/greathallX.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/greathallY.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/greathallZ.zip --directory-prefix=$dir
  (cd $dir;
  for i in *.zip; do
    unzip $i;
    rm -rf __*; # remove any files created by the zip binary
  done
  )

  #octagon
  #-------
  dir=c4dm/octagon/
  wget http://kakapo.dcs.qmul.ac.uk/irs/octagonOmni.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/octagonW.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/octagonX.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/octagonY.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/octagonZ.zip --directory-prefix=$dir
  (cd $dir;
  for i in *.zip; do
    unzip $i;
    rm -rf __*; # remove any files created by the zip binary
  done
  )

  #Classroom
  #----------
  dir=c4dm/classroom/
  wget http://kakapo.dcs.qmul.ac.uk/irs/classroomOmni.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/classroomW.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/classroomX.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/classroomY.zip --directory-prefix=$dir
  wget http://kakapo.dcs.qmul.ac.uk/irs/classroomZ.zip --directory-prefix=$dir
  (cd $dir;
  for i in *.zip; do
    unzip $i;
    rm -rf __*; # remove any files created by the zip binary
  done
  )
  )
fi

command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file

type_num=1
data_files=( $(find $RIR_home/c4dm/*/*/ -name '*.wav' -type f -print || exit -1) )
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files impulse responses in ${RIR_home}/c4dm/"
tmpdir=`mktemp -d $log_dir/c4dm_XXXXXX`
tmpdir=`utils/make_absolute.sh $tmpdir`
file_count=1
for data_file in ${data_files[@]}; do
  # c4dm has incompatible format of wav audio, which are not compatible with python's wav.read() function
 # output_file_name=${DBname}_type${type_num}_${file_count}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  output_file_name=${DBname}_type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  #echo "python local/multi_condition/read_rir.py --output-sampling-rate $sampling_rate wav ${tmpdir}/${file_count}.wav ${output_dir}/${output_file_name} || exit -1;" >> $command_file
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
