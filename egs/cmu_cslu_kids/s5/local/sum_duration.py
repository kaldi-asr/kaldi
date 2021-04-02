# Sum duration obtained by using 
# utils/data/get_utt2dur.sh

import sys
file = sys.argv[1]
sum = 0
with open(file, 'r') as fp:
    line = fp.readline()
    while(line):
        toks = line.strip().split()
        sum += float(toks[1])
        line = fp.readline()
fp.close()
h=sum/3600
sys.stdout.write("%f hour data.\n"%h)
