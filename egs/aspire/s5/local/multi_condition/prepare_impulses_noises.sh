#!/bin/bash
# set -e

# Copyright 2014  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0.
# This script processes RIRs available from available databases into 
# 8Khz wav files so that these can be used to corrupt the Fisher data.
# The databases used are:
# RWCP :  http://research.nii.ac.jp/src/en/RWCP-SSD.html (this data is mirrored @ openslr.org. Thanks to Mitsubishi Electric Research Laboratories)
# AIRD : Aachen Impulse response database (http://www.ind.rwth-aachen.de/en/research/tools-downloads/aachen-impulse-response-database/)
# Reverb2014 : http://reverb2014.dereverberation.com/download.html
# OpenAIR : http://www.openairlib.net/auralizationdb
# MARDY : http://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/
# QMUL impulse response dataset : http://c4dm.eecs.qmul.ac.uk/rdr/handle/123456789/6
# Impulse responses from Varechoic chamber at Bell Labs : http://www1.icsi.berkeley.edu/Speech/papers/gelbart-ms/pointers/
# Concert Hall impulse responses, Aalto University : http://legacy.spa.aalto.fi/projects/poririrs/
 
stage=0 
download_rirs=true # download the RIRs
sampling_rate=8000 # sampling rate to be used for the RIRs
log_dir=log # directory to store the log files
RIR_home=db/RIR_databases/ # parent directory of the RIR databases files
db_string="'aalto' 'air' 'rwcp' 'rvb2014' 'c4dm' 'varechoic' 'mardy' 'openair'" # RIR dbs to be used in the experiment

. cmd.sh
. path.sh
. utils/parse_options.sh

echo $*
if [ $# -ne 1 ]; then
  echo "$0 outputdir"
  exit 1;
fi

output_dir=$1
mkdir -p $output_dir/info ${output_dir}_non_normalized/info
mkdir -p $log_dir
rm -f $log_dir/type*.list

if [ -z "$db_string" ]; then
  echo "$0 : Please specify the db_string.";
  exit 1;
fi

# write the file_splitter to create job files for queue.pl
# we use this to parallelize the audio corruption and download jobs
cat << EOF > $log_dir/file_splitter.py
#!/usr/bin/env python
import os.path, sys, math

num_lines_per_file = int(sys.argv[1])
input_file = sys.argv[2]
[file_base_name, ext] = os.path.splitext(input_file)
lines = open(input_file).readlines();
num_lines = len(lines)
num_jobs = int(math.ceil(num_lines/ float(num_lines_per_file)))

# filtering commands into seperate task files
for i in xrange(1, num_jobs+1) :
  cur_lines = map(lambda index: lines[index], range(i - 1, num_lines , num_jobs))
  file = open("{0}.{1}{2}".format(file_base_name, i, ext), 'w')
  file.write("which python\n")
  file.write("".join(cur_lines))
  file.close()
print num_jobs
EOF
chmod +x $log_dir/file_splitter.py

if [ $stage -le 1 ]; then
  echo "Extracting the impulse responses from the databases $db_string"
  num_db_jobs=`echo $db_string|wc -w`
  $decode_cmd JOB=1:$num_db_jobs $log_dir/log/DBprocess.JOB.log \
    db=\(0 $db_string \)  \&\& \
    local/multi_condition/rirs/prep_\$\{db\[JOB\]\}.sh --file-splitter "$log_dir/file_splitter.py 10 " --download $download_rirs --sampling-rate $sampling_rate $RIR_home ${output_dir}_non_normalized $log_dir ||exit 1;
fi

if [ $stage -le 2 ]; then
  echo "Normalizing the extracted room impulse responses and noises, per type"
  echo "Note: Due to wav-format mismatch between sox and scipy, there might be warnings generated during file normalization."
  echo "      'WavFileWarning: Unknown wave file format' warnings are benign."
  # normalizing the RIR files 
  for i in `ls $log_dir/*type*.rir.list`; do
    echo "Processing files in $i"
    python local/multi_condition/normalize_wavs.py --is-room-impulse-response true $i $i.normval || exit 1;
    norm_coefficient=`cat $i.normval`
    echo "" > $i.normalized
    while read file_name; do
      if [ ! -z $file_name ]; then
        output_file_name=${output_dir}/`basename $file_name`
        sox --volume $norm_coefficient -t wav $file_name -t wav $output_file_name 2>/dev/null
        echo $output_file_name >> $i.normalized
      fi
    done < $i
  done

  # normalizing the noise files  
  for i in `ls $log_dir/*type*.noise.list`; do
    echo "Processing files in $i"
    python local/multi_condition/normalize_wavs.py --is-room-impulse-response false $i $i.normval || exit 1;
    norm_coefficient=`cat $i.normval`
    echo "" > $i.normalized
    while read file_name; do
      if [ ! -z $file_name ]; then
        output_file_name=${output_dir}/`basename $file_name`
        sox --volume $norm_coefficient -t wav $file_name -t wav $output_file_name 2>/dev/null
        echo $output_file_name >> $i.normalized
      fi
    done < $i
  done
fi

# copying the noise-rir pairing files
cp ${output_dir}_non_normalized/info/* $output_dir/info

# rename file location in the noise-rir pairing files 
for file in `ls $output_dir/info/noise_impulse*`; do
  sed -i "s/_non_normalized//g" $file
done

# generating the rir-list with probabilities alloted for each rir
db_string_python=$(echo $db_string|sed -e "s/'\s\+'/','/g")
python -c "
import glob, string, re
dbs=[$db_string_python]
rirs = []
for db in dbs:
  files = glob.glob('$log_dir/{0}*type*.rir.list.normalized'.format(string.upper(db)))
  for file in files:
    for line in open(file).readlines():
      if len(line.strip()) > 0:
        rirs.append(line.strip())
final_rir_list_file = open('$output_dir/info/impulse_files', 'w')
final_rir_list_file.write('\n'.join(rirs))
final_rir_list_file.close()
"

wc -l  $output_dir/info/impulse_files
