#!/usr/bin/env python


# Copyright 2018    Armin Oliya
# Apache 2.0.


# This script takes an existing ARPA lanugage model and limits the <unk> history 
# to make it suitable for downstream <unk> modeling. This is for the case when
# you don't have access to the original text corpus that is used for creating the LM.
# If you do, you can use pocolm with the option --limit-unk-history=true. This
# keeps the graph compact after adding the unk model.



import argparse
import codecs 
import os
import re 
from collections import defaultdict



parser = argparse.ArgumentParser(description='This script takes an existing ARPA lanugage model and limits the <unk> history to make it suitable for downstream <unk> modeling.')
parser.add_argument('--old_lm',required=True, help='path to the old ARPA language model', type=str)
parser.add_argument('--new_lm',required=True, help='path to the new ARPA model with limited unk history', type=str)

args = parser.parse_args()	




def get_ngram_stats(old_lm_lines):
    ngram_counts = defaultdict(int) 

    for i in range(10):
        g = re.search(r"ngram (\d)=(\d+)",old_lm_lines[i])
        if g:
            ngram_counts[int(g.group(1))] = int(g.group(2))
    
    max_ngrams = list(ngram_counts.keys())[-1]
    skip_rows = ngram_counts[1] 

    return max_ngrams, skip_rows, ngram_counts


def find_n_replace_unks(old_lm_lines,max_ngrams,skip_rows):

    new_lm_lines = old_lm_lines[:skip_rows]
    ngram_diffs =  defaultdict(int)

    unk_pattern = re.compile("[0-9.-]+(?:[\s\\t]\S+){1,3}[\s\\t]<unk>[\s\\t](?!-[0-9]+\.[0-9]+).*")
    backoff_pattern = re.compile("[0-9.-]+(?:[\s\\t]\S+){1,3}[\s\\t]<unk>[\s\\t]-[0-9]+\.[0-9]+")

    passed_2grams=False
    last_ngram=False

    unk_row_count = 0
    backoff_row_count = 0




    print "Upading the language model .. "



    for i in range(skip_rows, len(old_lm_lines)):

            l = old_lm_lines[i].strip()

        
            if "\{}-grams:".format(3) in l:
                passed_2grams = True
            if "\{}-grams:".format(max_ngrams) in l:
                last_ngram = True
                
                
            # remove any n-gram states of the form: foo <unk> -> X 
            # that is, any n-grams of order > 2 where <unk> is the second-to-last word
            # here we skip 1-gram and 2-gram sections of arpa 

            if passed_2grams:
                g_unk = unk_pattern.search(l)
                if g_unk:
                    ngram = len(g_unk.group(0).split()) - 1 
                    ngram_diffs[ngram] = ngram_diffs[ngram] - 1
                    unk_row_count += 1
                    continue 

            
            # remove backoff probability from the lines that end with <unk>
            # for example, the -0.64 in -4.09 every <unk> -0.64
            # here we skip the last n-gram section because it doesn't include backoff probabilities

            if not last_ngram:  
                g_backoff = backoff_pattern.search(l)
                if g_backoff:
                    updated_row = g_backoff.group(0).split()[:-1]
                    updated_row = updated_row[0]+"\t"+(" ".join(updated_row[1:]))+"\n"
                    new_lm_lines.append(updated_row)
                    backoff_row_count += 1
                    continue
                
            new_lm_lines.append(l+"\n")


    print ("Removed {} lines including <unk> as second-to-last term".format(unk_row_count))
    print ("Removed backoff probabilties from {} lines".format(backoff_row_count))

    return new_lm_lines, ngram_diffs



def write_new_lm(new_lm_lines,ngram_counts,ngram_diffs):
    # update n-gram counts that go in the header of the arpa lm


    for i in range(10):
        
        g = re.search(r"ngram (\d)=(\d+)",new_lm_lines[i])
        if g:
            n = int(g.group(1))
            if n in ngram_diffs:
                new_num_ngrams = ngram_counts[n] - ngram_diffs[n]
                new_lm_lines[i]="ngram {}={}\n".format(n,new_num_ngrams)
    
        
    with codecs.open(os.path.expanduser(args.new_lm),"w", encoding="utf-8") as f: 
        f.writelines(new_lm_lines)




def main(): 


    with codecs.open(os.path.expanduser(args.old_lm),"r", encoding="utf-8") as f: 
        old_lm_lines = f.readlines()

    max_ngrams, skip_rows,  ngram_counts = get_ngram_stats(old_lm_lines)

    new_lm_lines, ngram_diffs = find_n_replace_unks(old_lm_lines,max_ngrams,skip_rows)

    write_new_lm(new_lm_lines,ngram_counts,ngram_diffs)



if __name__ == "__main__":
    main()