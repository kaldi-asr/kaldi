#!/bin/bash
# Copyright 2016  Tom Ko
# Apache 2.0
# script to simulate room impulse responses (RIRs)

sampling_rate=8000
output_bit=16
num_room=30
rir_per_room=2

. cmd.sh
. path.sh
. ./utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 [options] <output-dir>"
  echo "e.g.:"
  echo " $0  data/impulses_noises"
  exit 1;
fi

output_dir=$1
mkdir -p $output_dir

cat >./genrir.m <<EOF
path(path,'local/reverberate')
mex local/reverberate/rir_generator.cpp
c = 340;                    % Sound velocity (m/s)
fs = $sampling_rate;        % Sample frequency (samples/s)
num_sample = 2048;          % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = 2;                  % Reflection order
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter
BitsPerSample = $output_bit;
Rconstant=24 * log(10.0) / c;
Room_bound_x = [1 15];      % upper and lower bound of the room size in sampling
Room_bound_y = [1 15];
Room_bound_z = [2 5];
T60_bound = [0.03 2];       % upper and lower bound of the reverberation time in sampling
maxRS=5;                    % max distance between the speaker and receiver
dirname = '$output_dir/';
file_rir_info = fopen(strcat(dirname, 'rir_list'),'w');
file_room_info = fopen(strcat(dirname, 'room_list'),'w');
for room_id = 1 : $num_room
    room_x = round(rand * (Room_bound_x(2)-Room_bound_x(1)) + Room_bound_x(1), 2);
    room_y = round(rand * (Room_bound_y(2)-Room_bound_y(1)) + Room_bound_y(1), 2);
    room_z = round(rand * (Room_bound_z(2)-Room_bound_z(1)) + Room_bound_z(1), 2);
    Room_xyz = [room_x room_y room_z]          % Room dimensions [x y z] (m)
    Mic_xyz = [rand*Room_xyz(1) rand*Room_xyz(2) rand*Room_xyz(3)]    % Receiver position [x y z] (m)
    fprintf(file_room_info,'Room%03d %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f\n', room_id, Room_xyz, Mic_xyz);
    for rir_id = 1 : $rir_per_room
        t60 = round(rand * (T60_bound(2)-T60_bound(1)) + T60_bound(1), 2);
        Source_bound_x = [max(Mic_xyz(1)-maxRS, 0) min(Mic_xyz(1)+maxRS, Room_xyz(1))];
        Source_bound_y = [max(Mic_xyz(2)-maxRS, 0) min(Mic_xyz(2)+maxRS, Room_xyz(2))];
        Source_bound_z = [max(Mic_xyz(3)-maxRS, 0) min(Mic_xyz(3)+maxRS, Room_xyz(3))];

        while 1
            source_x = rand * (Source_bound_x(2)-Source_bound_x(1)) + Source_bound_x(1);
            source_y = rand * (Source_bound_y(2)-Source_bound_y(1)) + Source_bound_y(1);
            source_z = rand * (Source_bound_z(2)-Source_bound_z(1)) + Source_bound_z(1);
            Source_xyz = [source_x source_y source_z];          % Source position [x y z] (m)
            if norm(Source_xyz - Mic_xyz) < maxRS
                break
            end
        end
        Volume = Room_xyz(1) * Room_xyz(2) * Room_xyz(3);
        Surface = 2 * (Room_xyz(1) * Room_xyz(2) + Room_xyz(2) * Room_xyz(3) + Room_xyz(3) * Room_xyz(1));
        minrt60 = Rconstant * Volume / Surface / 0.99;
        t60 = max(t60, minrt60);           % Reverberation time
        rir = rir_generator(c, fs, Mic_xyz, Source_xyz, Room_xyz, t60, num_sample, mtype, order, dim, orientation, hp_filter);
        [RT,DRR,CTE,CFS,EDT] = IR_stats(rir', fs);
        rir_filename = strcat(dirname,'matlab_', num2str(round(t60, 2)), '_', num2str(round(CTE, 2)), '.wav');
        audiowrite(rir_filename, rir, fs, 'BitsPerSample', BitsPerSample);
        fprintf(file_rir_info,'Room%03d-%05d simulated %s\n', room_id, rir_id, rir_filename);
    end
end
fclose(file_rir_info);
fclose(file_room_info);
EOF
matlab -nosplash -nodesktop < ./genrir.m
rm genrir.m rir_generator.mexa64

