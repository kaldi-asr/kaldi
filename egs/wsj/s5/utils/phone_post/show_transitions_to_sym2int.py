#!/usr/bin/python

import warnings
import sys, os
import time
import re


def show_transitions_to_sym2int(show_transitions_file, roots_txt_file):

    # read roots.txt file 
    roots_txt = open(roots_txt_file)
    roots_dict = dict()
    for line in roots_txt:
        line = line.rstrip()
        
        r=line.split()[2]
        roots_dict[r]=r
        for l in line.split()[3:]:
            if l in roots_dict.keys():
                print "Error: leaf associated with multiple roots"
                sys.exit(1)
            else:
                roots_dict[l]=r
	#	print "root is "+r+ " , leaf is "+l

    show_transitions = open(show_transitions_file)
    for line in show_transitions:
        line = line.rstrip()
        if re.search('pdf', line):
            pdf=line.split(" ")[10]
            this_leaf=line.split(" ")[4]
            
            print pdf, roots_dict[this_leaf]



if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print "Error: Wrong number of input arguments"
        print "Usage: "+sys.argv[0]+" show_transitions.txt roots.txt"
        print "prints to_sym2int to stdout"
        sys.exit(1);


    show_transitions_file=sys.argv[1]
    roots_txt_file=sys.argv[2]

    show_transitions_to_sym2int(show_transitions_file, roots_txt_file)



