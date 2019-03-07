#!/usr/bin/env python3

# 主要功能是将aishell1的transcript文件中的说话人编号BAC009S0002W0122 转换成AISHELL2格式 IS0002W0122
# 因为aishell2采用结巴分词，移除标注的空格

new_transcripts = []
new_wav_scp = []


aishell_transcripts = open("../transcript/aishell_transcript_v0.8_remove_space.txt", encoding="utf-8")

transcripts = aishell_transcripts.readlines()

trans_txt = open("train/trans.txt", 'w', encoding="utf-8")
wav_scp = open("train/wav.scp", 'w', encoding="utf-8")


for transcript in transcripts:
    print(transcript)
    spkid = "I" + transcript[6:16]
    print(spkid)
    lable = transcript[16:len(transcript)]
    print(lable)
    new_transcripts.append(spkid + "\t" + lable)
    new_wav_scp.append(spkid + "\t" + "wav/"+spkid[1:6]+"/"+spkid+".wav\n")

print(new_transcripts)
trans_txt.writelines(new_transcripts)
wav_scp.writelines(new_wav_scp)

aishell_transcripts.close()
trans_txt.close()
wav_scp.close()

print("All Done!")

