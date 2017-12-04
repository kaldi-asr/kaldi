#!/bin/sh
#Argument -> njobs enhancement_method enhanced_directory destination_directory chime_RIR_directory

njobs=$1
enhancement_method=$2
enhancement_directory=$3
destination_directory=$4
chime_RIR_directory=$5
ls $chime_RIR_directory/dt05_*/*CH5.Clean.wav > original_list
ls $enhancement_directory/dt05_*simu/*.wav > enhanced_list
matlab -nodisplay -nosplash -r "addpath('local'); stoi_estoi_sdr($njobs,'$enhancement_method','$destination_directory','dt05');exit"
ls $chime_RIR_directory/et05_*/*CH5.Clean.wav > original_list
ls $enhancement_directory/et05_*simu/*.wav > enhanced_list
matlab -nodisplay -nosplash -r "addpath('local'); stoi_estoi_sdr($njobs,'$enhancement_method','$destination_directory','et05');exit"
rm original_list enhanced_list
