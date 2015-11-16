#!/bin/bash

# Copyright 2015 Hakan Erdogan


echo This script runs channel adaptation of some wav files
echo channel adaptation means finding an FIR filter such that when the first file is filtered with this FIR filter
echo the difference between the filtered first file and the second file is minimized.
echo In this example, the first list of files are close talking microphone \(CH0\) files from Chime3 database.
echo The second list of files are distant microphone \(CH4 or CH5\) or artifically modified versions of the CH0 files
echo Modification involves reverberation and possibly additional noise
echo The RMSE\(1,2\) between first and second files are high due to reverb and possibly noise.
echo After channel adapting the first one and obtaining the third set of files, RMSE\(3,2\) \< RMSE\(1,2\) must be true
echo The reason is that the filtering will get rid of reverberation effects, but not noise.
echo The modified second file of M05_440C0203b_BUS is obtained by filtering the first file using a 50-tap 0-centered filter
echo The modified second file of M05_440C0203c_BUS is obtained by delaying the first file by 50 samples
echo The modified second file of M05_440C0203d_BUS is obtained by advancing the first file by 50 samples
echo So: for M05_440C0203b_BUS M05_440C0203c_BUS M05_440C0203d_BUS, we expect the RMSE\(3,2\) to be very small due to filtering
echo operations which should be recovered easily using channel adaptation with appropriate number of taps.
echo Note that the taps are for an FIR filter that is centered at 0, so at least 100 taps are required to correct for 50 samples advance and delay

conf=/tmp/stft_$$.conf
wav1=/tmp/wavclean_$$.scp
wav2=/tmp/wavnoisy_$$.scp
wav3=/tmp/wavadapt_$$.scp
wavtgz=wavs_test_channeladapt.tgz

tar xzvf $wavtgz

cat << EOF > $wav1
F05_440C020E_BUS wav/F05_440C020E_BUS.CH0.wav
F06_440C020A_BUS wav/F06_440C020A_BUS.CH0.wav
M05_440C0203a_BUS wav/M05_440C0203_BUS.CH0.wav
M05_440C0203b_BUS wav/M05_440C0203_BUS.CH0.wav
M05_440C0203c_BUS wav/M05_440C0203_BUS.CH0.wav
M05_440C0203d_BUS wav/M05_440C0203_BUS.CH0.wav
M06_440C020N_BUS wav/M06_440C020N_BUS.CH0.wav
EOF

cat << EOF > $wav2
F05_440C020E_BUS wav/F05_440C020E_BUS.CH5.wav
F06_440C020A_BUS wav/F06_440C020A_BUS.CH4.wav
M05_440C0203a_BUS wav/M05_440C0203_BUS.CH5.wav
M05_440C0203b_BUS wav/M05_440C0203_BUS.CH0_filtered.wav
M05_440C0203c_BUS wav/M05_440C0203_BUS.CH0_delayed.wav
M05_440C0203d_BUS wav/M05_440C0203_BUS.CH0_advanced.wav
M06_440C020N_BUS wav/M06_440C020N_BUS.CH5.wav
EOF

cat << EOF > $wav3
F05_440C020E_BUS wav/F05_440C020E_BUS.CH0_ca5.wav
F06_440C020A_BUS wav/F06_440C020A_BUS.CH0_ca4.wav
M05_440C0203a_BUS wav/M05_440C0203_BUS.CH0_ca5.wav
M05_440C0203b_BUS wav/M05_440C0203_BUS.CH0_cafiltered.wav
M05_440C0203c_BUS wav/M05_440C0203_BUS.CH0_cadelayed.wav
M05_440C0203d_BUS wav/M05_440C0203_BUS.CH0_caadvanced.wav
M06_440C020N_BUS wav/M06_440C020N_BUS.CH0_ca5.wav
EOF

echo "--80 taps------"
../../featbin/channel-adapt --taps=80 scp:$wav1 scp:$wav2 scp:$wav3
echo "--100 taps------"
../../featbin/channel-adapt --taps=100 scp:$wav1 scp:$wav2 scp:$wav3
echo "--101 taps------"
../../featbin/channel-adapt --taps=101 scp:$wav1 scp:$wav2 scp:$wav3
echo "--200 taps------"
../../featbin/channel-adapt --taps=200 scp:$wav1 scp:$wav2 scp:$wav3
echo "--800 taps------"
../../featbin/channel-adapt --taps=800 scp:$wav1 scp:$wav2 scp:$wav3
echo "--------"
