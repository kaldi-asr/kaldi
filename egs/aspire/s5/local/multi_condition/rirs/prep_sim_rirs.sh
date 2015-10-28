#!/bin/bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the impulse responses and noise files from the
# Reverb2014 challenge
# and converts them to wav files with the required sampling rate
#==============================================

download=true
sampling_rate=8000
output_bit=16
DBname=SIM_RIRS
RIR_total=75000
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

cat >$log_dir/genrir.m <<EOF
path(path,'local/multi_condition/rirs')
mex local/multi_condition/rirs/rir_generator.cpp
c = 340;                    % Sound velocity (m/s)
fs = $sampling_rate;        % Sample frequency (samples/s)
R = [2 1.5 2];              % Receiver position [x y z] (m)
S = [2 3.5 2];              % Source position [x y z] (m)
L = [5 4 6];                % Room dimensions [x y z] (m)
t60 = 0.4;                  % Reverberation time
n = 2048;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = 2;                  % Reflection order
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter
BitsPerSample = $output_bit;
Rconstant=24 * log(10.0) / c;
x1=1; x2=15;                % upper and lower bound of the room size in sampling
y1=1; y2=15;
z1=2; z2=5;
t1=0.03; t2=2;              % upper and lower bound of the reverberation time in sampling
maxRS=5;                    % max distance between the speaker and receiver
%  minVolume = x2 * y2 * z2;
%  minSurface = 2 * (x2 * y2 + y2 * z2 + x2 * z2);
%  t1_lowerbound = Rconstant * minVolume / minSurface
%  t1=t1_lowerbound;
for i = 1 : $RIR_total
  room_x = round(rand * (x2-x1) + x1, 2);
  room_y = round(rand * (y2-y1) + y1, 2);
  room_z = round(rand * (z2-z1) + z1, 2);
  t60 = round(rand * (t2-t1) + t1, 2);
  L = [room_x room_y room_z];
  R = [rand*L(1) rand*L(2) rand*L(3)];
  sr_x1 = min(R(1)+maxRS, L(1));
  sr_y1 = min(R(2)+maxRS, L(2));
  sr_z1 = min(R(3)+maxRS, L(3));
  sr_x2 = max(R(1)-maxRS, 0);
  sr_y2 = max(R(2)-maxRS, 0);
  sr_z2 = max(R(3)-maxRS, 0);

  while 1
    sx = rand * (sr_x2 - sr_x1) + sr_x1;
    sy = rand * (sr_y2 - sr_y1) + sr_y1;
    sz = rand * (sr_z2 - sr_z1) + sr_z1;
    S = [sx sy sz];
    if norm(S-R) < maxRS
      break
    end
  end
%L
%t60
  Volume = L(1) * L(2) * L(3);
  Surface = 2 * (L(1) * L(2) + L(2) * L(3) + L(3) * L(1));
  minrt60 = Rconstant * Volume / Surface / 0.99;
  t60 = max(t60, minrt60);
  rir = rir_generator(c, fs, R, S, L, t60, n, mtype, order, dim, orientation, hp_filter);
  [RT,DRR,CTE,CFS,EDT] = IR_stats(rir', fs);
  dirname = '$output_dir/';
  filename = strcat(dirname,'matlab_', num2str(round(t60, 2)), '_', num2str(round(CTE, 2)), '.wav')
  audiowrite(filename, rir, fs, 'BitsPerSample', BitsPerSample);
end
EOF
matlab -nosplash -nodesktop < $log_dir/genrir.m
rm rir_generator.mexa64

type_num=1
data_files=( $(find $output_dir -name 'matlab*.wav' -type f -print || exit -1) )
files_done=0
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files simulated impulse responses in $output_dir."
for data_file in ${data_files[@]}; do
  output_file_name=${DBname}_type${type_num}_`basename $data_file | tr '[:upper:]' '[:lower:]'`
  mv $data_file ${output_dir}/${output_file_name}
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.rir.list
  files_done=$((files_done + 1))
done

