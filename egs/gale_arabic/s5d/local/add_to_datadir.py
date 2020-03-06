#!/usr/bin/python

# This script appends utterances dumped out from XML to a Kaldi datadir

import sys, re
from xml.sax.saxutils import unescape

basename=sys.argv[1]
outdir = sys.argv[2]

if len(sys.argv) > 3:
    mer_thresh=float(sys.argv[3])
else:
    mer_thresh = None

# open the output files in append mode
#segments_file = open(outdir + '/segments', 'a')
#utt2spk_file = open(outdir + '/utt2spk', 'a')
#text_file = open(outdir + '/text', 'a')
mgb2_file = open(outdir + '/mgb2', 'a')

for line in sys.stdin:

    m = re.match(r'\w+speaker(\d+)\w+\s+(.*)', line)
    #print line

    if m:

        spk = int(m.group(1))

        t = m.group(2).split()
        start = float(t[0])
        end = float(t[1])
        mer = float(t[2])
        
        s = [unescape(w) for w in t[3:]]       
        words = ' '.join(s)

        segId = '%s_spk-%04d_seg-%07d:%07d' % (basename, spk, start*100, end*100)
        spkId = '%s_spk-%04d' % (basename, spk)

        # only add segments where the Matching Error Rate is below the prescribed threshhold
        if mer_thresh == None or mer <= mer_thresh:
#print >> segments_file, '%s %s %.2f %.2f' % (segId, basename, start, end ) 
#print >> text_file, '%s %s' % (segId, words)
#print >> utt2spk_file, '%s %s' % (segId, spkId)
            print >> mgb2_file, '%s %s %.3f %.3f %s' % (basename, segId, start, end, words)

#segments_file.close()
#utt2spk_file.close()
#text_file.close()
mgb2_file.close()
 
            
