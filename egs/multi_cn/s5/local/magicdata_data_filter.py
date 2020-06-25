#!/usr/bin/env python3
import os
import sys

def main(argv):
  try:
    files={}
    for line in open(argv[0]):
      fpath = line.strip('\r\n')
      wname = os.path.basename(fpath)
      files[wname] = fpath
  except IOError:
    print(argv[0] + " not exist!")
    sys.exit(1)

  bad = []
  if len(argv) == 4:
    for line in open(argv[3]):
      bad.append(line.strip('\r\n'))

  fWavScp = open(os.path.join(argv[2], 'wav.scp'), 'w')
  fText = open(os.path.join(argv[2], 'transcripts.txt'), 'w', encoding = "utf-8")
  fUtt2Spk = open(os.path.join(argv[2], 'utt2spk'), 'w')
  for line in open(argv[1], encoding = "utf-8"):
    if '.wav' not in line:
      continue
    (wavid, spkid, text) = line.strip('\r\n').split('\t')
    if len(bad) > 0 and wavid in bad:
      continue
    if wavid in files.keys():
      uttid = wavid.replace('.wav', '')
      fWavScp.write(uttid + ' ' + files[wavid] + '\n')
      fText.write(uttid + ' ' + text + '\n')
      fUtt2Spk.write(uttid + ' ' + spkid + '\n')
  fWavScp.close()
  fText.close()
  fUtt2Spk.close()

if __name__ == "__main__":
  main(sys.argv[1:])
