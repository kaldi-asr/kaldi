#!/usr/bin/python

# Author: Hendrik Meutzner
# Version: 1.0 (2016/05/31)

import sys, getopt, re
from collections import Counter

def compute_scores(filename):

    digit_map = {"ZERO": "z", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}

    letter_scores = Counter()
    digit_scores = Counter()
    sent_scores = Counter()
    letter_digit_scores = Counter()
    num_conditions = Counter()
    
    with open(filename, 'r') as fid:
        data = [l.strip() for l in fid.readlines()]

    for d in data:
    
        tmp_all = d.split()
        tmp_utt = tmp_all[0].split('_')
        
        utt_id = '_'.join(tmp_utt[:2])      # speaker_utterance-id
        cond = tmp_utt[2]                   # condition (e.g., 0dB, clean, etc.)
        ref_trans = tmp_utt[1]
        
        # fetch and shorten recognized transcription, e.g., "BIN BLUE BY F TWO NOW" -> "B B B F 4 N"
        rec_trans = tmp_all[1:]
        
        if len(rec_trans) > 5:
            rec_trans[4] = digit_map[rec_trans[4]]
            
        rec_trans = ''.join([r[0].lower() for r in rec_trans])

        #print "[%s] %s: %s" %(cond, utt_id, rec_trans)
        
        num_conditions[cond] += 1
        
        if len(rec_trans) > 3:
            if rec_trans[3] == ref_trans[3]:
                letter_scores[cond] += 1
                
        if len(rec_trans) > 4:
            if rec_trans[4] == ref_trans[4]:
                digit_scores[cond] += 1
        
        if len(rec_trans) == 6:
            if rec_trans == ref_trans:
                sent_scores[cond] += 1    
    
    
    for (kl, vl), (kd, vd), (ks, vs) in zip(letter_scores.items(), digit_scores.items(), sent_scores.items()):
        
        letter_scores[kl] = float(vl)/float(num_conditions[kl])
        digit_scores[kd] = float(vd)/float(num_conditions[kd])
        letter_digit_scores[kl] = float(vl+vd)/float(2*num_conditions[kl])
        sent_scores[ks] = float(vs)/float(num_conditions[ks])
     
    return (letter_scores, digit_scores, letter_digit_scores, sent_scores)
     
def print_scores(scores_all, cond_list_manual=[]):

    """ We want this:
    Keyword (letter+digit) recognition accuracy (%)
    -----------------------------------------------------------------
    SNR       -6dB    -3dB    0dB     3dB     6dB     9dB     Average
    -----------------------------------------------------------------
    Overall   40.69   41.21   51.98   62.24   72.93   79.83   58.15
    -----------------------------------------------------------------
    Letter    33.10   31.21   39.31   49.31   60.34   68.79   47.01
    Digit     48.28   51.21   64.66   75.17   85.52   90.86   69.28
    -----------------------------------------------------------------
    """

    (letter_scores, digit_scores, letter_digit_scores, sent_scores) = scores_all
    
    if cond_list_manual:
        cond_list = cond_list_manual.split(',')
    else:
        cond_list = sorted(letter_scores.keys())
    
    SEP_LINE = "-----------------------------------------------------------------\n"
        
    # Top and header
    outStr = "Keyword (letter+digit) recognition accuracy (%)\n"
    outStr += SEP_LINE
    outStr += "%-10s" %("SNR/Type")
    for k in cond_list:
        
        # replace the "m" in m6dB and m3dB by "-"
        k_new = re.sub(r'^m(\ddB)$', r'-\1', k, flags=re.IGNORECASE)
        
        outStr += "%-8s" %(k_new)
        
    outStr += "Average\n"
    outStr += SEP_LINE    
    
    # Overall block
    outStr += "%-10s" %("Overall")
    for k in cond_list:
        #outStr += "%-6.2f" %(100*letter_digit_scores[k])
        outStr += "{:<8.2f}".format(100*letter_digit_scores[k])
    outStr += "{:<8.2f}\n".format(100*sum(letter_digit_scores.values())/len(letter_digit_scores.values()))
    outStr += SEP_LINE    
    
    # Letter+Digit block
    outStr += "%-10s" %("Letter")
    for k in cond_list:
        outStr += "{:<8.2f}".format(100*letter_scores[k])
    outStr += "{:<8.2f}\n".format(100*sum(letter_scores.values())/len(letter_scores.values()))
    outStr += "%-10s" %("Digit")
    for k in cond_list:
        outStr += "{:<8.2f}".format(100*digit_scores[k])
    outStr += "{:<8.2f}\n".format(100*sum(digit_scores.values())/len(digit_scores.values()))
    outStr += SEP_LINE
    
    print outStr
    
    
def usage():
    print 'Usage: '+sys.argv[0]+' --ifile <inputfile> [--list <snr/type list>]'
    print 'Example 1: '+sys.argv[0]+' --ifile ./exp/tri1b/decode/scoring/trans.txt --list "m6dB,m3dB,0dB,3dB,6dB,9dB"'
    print 'Example 2: '+sys.argv[0]+' --ifile ./exp/tri1b/decode/scoring/trans.txt --list "clean,reverberated"'
    
def main(argv):

    inputfile = ''
    snr_type_list = "m6dB,m3dB,0dB,3dB,6dB,9dB"

    try:
        opts, args = getopt.getopt(argv,"hi:l",["help","ifile=","list="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-l", "--list"):
            snr_type_list = arg            

    scores_all = compute_scores(inputfile)    
    print_scores(scores_all, snr_type_list)
    
if __name__ == "__main__":
   main(sys.argv[1:])
