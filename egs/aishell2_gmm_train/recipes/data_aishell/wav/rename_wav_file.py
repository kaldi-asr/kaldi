

# 主要功能是将aishell1语料中的BAC009S0002W0122.wav 重命名成aishell2的格式 IS0002W0122.wav

import subprocess

(status, outputs) = subprocess.getstatusoutput('find train/wav/ -name *.wav')

wav_files = outputs.split("\n")

for wav_file in wav_files:
    print(wav_file)
    new_wav_file = wav_file.replace("BAC009","I")
    print(new_wav_file)
    args = 'mv ' + wav_file + " " + new_wav_file
    subprocess.getstatusoutput(args)

print ("All done!")
