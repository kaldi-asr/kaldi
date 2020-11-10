#!/usr/bin/env python3
import os
import sys
import json

def main(argv):
  fp = open(argv[1], encoding = "utf-8")
  js = json.load(fp)
  fp.close()
  metas = {}
  for ele in js:
    fname = ele['file']
    metas[fname] = ele

  fWavScp = open(os.path.join(argv[2], 'wav.scp'), 'w')
  fText = open(os.path.join(argv[2], 'transcripts.txt'), 'w', encoding = "utf-8")
  fUtt2Spk = open(os.path.join(argv[2], 'utt2spk'), 'w')
  for line in open(argv[0]):
    fpath = line.strip('\r\n')
    wname = os.path.basename(fpath)
    meta = metas[wname]
    spkid = 'P' + meta['user_id']
    uttid = spkid + '-' + meta['id']
    fWavScp.write(uttid + ' ' + fpath + '\n')
    fText.write(uttid + ' ' + meta['text'] + '\n')
    fUtt2Spk.write(uttid + ' ' + spkid + '\n')
  fWavScp.close()
  fText.close()
  fUtt2Spk.close()

if __name__ == "__main__":
  main(sys.argv[1:])
