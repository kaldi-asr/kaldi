import sys
import errno    
import os


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

#mapping

old_d=sys.argv[1]
new_d=sys.argv[2]

percond_map = {}
percond_map['0'] = "cleanxxxxx_wv1"
percond_map['1'] = "cleanxxxxx_wv2"
percond_map['2'] = "airportxxx_wv1"
percond_map['3'] = "airportxxx_wv2"
percond_map['4'] = "babblexxxx_wv1"
percond_map['5'] = "babblexxxx_wv2"
percond_map['6'] = "carxxxxxxx_wv1"
percond_map['7'] = "carxxxxxxx_wv2"
percond_map['8'] = "restaurant_wv1"
percond_map['9'] = "restaurant_wv2"
percond_map['a'] = "streetxxxx_wv1"
percond_map['b'] = "streetxxxx_wv2"
percond_map['c'] = "trainxxxxx_wv1"
percond_map['d'] = "trainxxxxx_wv2"


#utt2spk
lines = map(lambda x: x.strip(), open(old_d+"/utt2spk", "r").readlines())
new_utt2spk_lines=""
for line in lines:
  c=line.split()[0][-1]
  new_utt2spk_lines = new_utt2spk_lines+percond_map[c]+'_'+line.split()[0]+' '+percond_map[c]+'_'+line.split()[1]+'\n'
new_utt2spk_lines = new_utt2spk_lines.strip()

#wav.scp
lines = map(lambda x: x.strip(), open(old_d+"/wav.scp", "r").readlines())
new_wav_scp_lines=""
for line in lines:
  c=line.split()[0][-1]
  new_wav_scp_lines = new_wav_scp_lines+percond_map[c]+'_'+line.split()[0]+' '+' '.join(line.split()[1:])+'\n'
new_wav_scp_lines = new_wav_scp_lines.strip()

#text
lines = map(lambda x: x.strip(), open(old_d+"/text", "r").readlines())
new_text_lines=""
for line in lines:
  c=line.split()[0][-1]
  new_text_lines = new_text_lines+percond_map[c]+'_'+line.split()[0]+' '+' '.join(line.split()[1:])+'\n'
new_text_lines = new_text_lines.strip()


mkdir_p(new_d)
#print new_text_lines
fp = open(new_d+'/utt2spk', 'w')
fp.write(new_utt2spk_lines)
fp.close()

#print new_wav_scp_lines
fp = open(new_d+'/wav.scp', 'w')
fp.write(new_wav_scp_lines)
fp.close()

#print text
fp = open(new_d+'/text', 'w')
fp.write(new_text_lines)
fp.close()



