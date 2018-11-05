import sys, os

src_dir = sys.argv[1]
data_dir = sys.argv[2]
if not os.path.exists(data_dir):
  os.makedirs(data_dir)
wavscp_fi = open(data_dir + "/wav.scp" , 'w')
utt2spk_fi = open(data_dir + "/utt2spk" , 'w')
segments_fi = open(data_dir + "/segments" , 'w')
rttm_fi = open(data_dir + "/rttm" , 'w')
for subdir, dirs, files in os.walk(src_dir):
  for file in files:
    filename = os.path.join(subdir, file)
    if filename.endswith(".lab"):
      utt = os.path.basename(filename).split(".")[0]
      lines = open(filename, 'r').readlines()
      segment_id = 0
      for line in lines:
        start, end, speech = line.split()
        segment_id_str = utt + "_" + str(segment_id).zfill(4)
        segments_str = segment_id_str + " " + utt + " " + start + " " + end + "\n"
        utt2spk_str = segment_id_str + " " + utt + "\n"
        segments_fi.write(segments_str)
        utt2spk_fi.write(utt2spk_str)
        segment_id += 1
      wav_str = utt  + " sox -t flac " + src_dir + "/data/flac/" + utt + ".flac -t wav -r 16k -b 16 - channels 1 |\n"
      wavscp_fi.write(wav_str)
      rttm_str = open(src_dir + "/data/rttm/" + utt + ".rttm", 'r').read()
      rttm_fi.write(rttm_str)

segments_fi.close()
utt2spk_fi.close()
wavscp_fi.close()
rttm_fi.close()
