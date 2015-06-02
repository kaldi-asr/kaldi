#!/usr/bin/env python
'''
This script assumes that the parallel files have the same filename with different extensions and you must
specify the absolute path to the corpus from the root. The text files may only contain a single line of text.

'''

import sys, os, codecs

srcdir = sys.argv[1]
dest = sys.argv[2]
snd_ext = sys.argv[3]
txt_ext = sys.argv[4]

corpus = os.listdir(srcdir)

text = codecs.open(os.path.join(dest, "text"), "w", "utf8")
wavscp = codecs.open(os.path.join(dest, "wav.scp"), "w", "utf8")
utt2spk = codecs.open(os.path.join(dest, "utt2spk"), "w", "utf8")
sndlist = []
txtlist = []

for line in corpus:
    stem_and_ext = line.strip().rsplit(".", 1)
    if len(stem_and_ext) == 2:
        if stem_and_ext[-1] == snd_ext:
            sndlist.append(stem_and_ext[0])
        elif stem_and_ext[-1] == txt_ext:
            txtlist.append(stem_and_ext[0])

stems = sorted(list(set(sndlist) & set(txtlist)))

#print(stems)

# Use the filename as utterance id

for uttid in stems:
    fin = uttid+ "." +txt_ext
    utt = codecs.open(os.path.join(srcdir, fin), "r", "utf8").read()
    text.write(uttid+ " " +utt)
    spkid = uttid.rsplit("_")[0]
    wavscp.write(uttid+ " " +os.path.join(srcdir, uttid+ "." +snd_ext)+ "\n")
    utt2spk.write(uttid+ " " +spkid+ "\n")
    
utt2spk.close()
text.close()
wavscp.close()
    
