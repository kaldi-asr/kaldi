#!/usr/bin/python

import warnings
import sys, os
import time


def pseudo_phones(roots_txt_file):

    with open(roots_txt_file) as f:
        roots_txt = f.read().splitlines()

    cnt=0
    for roots in roots_txt:
        this_root=roots.split()[2]
        print this_root, str(cnt)


        # if list(this_root)[0] != "_":
        #     print str(cnt)+" "+this_root.split("_")[0]
        # else:
        #     print str(cnt)+" "+"_"+this_root.split("_")[1]
        
        cnt=cnt+1

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print "Error: Wrong number of input arguments"
        print "Usage: "+sys.argv[0]+" lang/phones/roots.txt"
        print "prints pseudo_phones.txt to stdout"
        sys.exit(1);

    roots_txt_file=sys.argv[1]
    roots_basename=os.path.basename(roots_txt_file)

    if roots_basename != "roots.txt":
        warnings.warn("given roots_txt_file="+roots_txt_file+" is not usual kaldi roots.txt, proceeding ...")
        time.sleep(10)
    

    pseudo_phones(roots_txt_file)

    

