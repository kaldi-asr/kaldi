#!/usr/bin/env python
# convert acronyms in swbd decode result to fisher convention
# e.g. convert things like en_4156 B 414.26 0.65 u._c._l._a. to
# en_4156 B 414.26 0.16 u
# en_4156 B 414.42 0.16 c
# en_4156 B 414.58 0.16 l
# en_4156 B 414.74 0.17 a

import argparse,re
__author__ = 'Minhua Wu'
 
parser = argparse.ArgumentParser(description='format acronyms from a._b._c. to a b c')
parser.add_argument('-i','--input', help='Input ctm file ',required=True)
parser.add_argument('-o','--output',help='Output ctm file', required=True)
args = parser.parse_args()

fin = open(args.input,"r")
fout = open(args.output, "w")

for line in fin:
    items = line.split()
    
    if items[4].find(".") != -1:
        letters = items[4].split("._")
        acronym_period = round(float(items[3]), 2)
        letter_slot = round(acronym_period / len(letters), 2)
        time_start = round(float(items[2]), 2)
        for l in letters[:-1]:
            time = " %.2f %.2f " % (time_start, letter_slot)
            fout.write(' '.join(items[:2])+ time + l + ".\n")
            time_start = time_start + letter_slot
        last_slot = acronym_period - letter_slot * (len(letters) - 1)
        time = " %.2f %.2f " % (time_start, last_slot)
        letters[-1] = re.sub(r"\.'s", "'s", letters[-1])
        letters[-1] = re.sub(r"\.s", "'s", letters[-1])        
        fout.write(' '.join(items[:2])+ time + letters[-1] + "\n")
    else:
        fout.write(line)    


