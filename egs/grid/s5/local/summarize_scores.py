#!/bin/python

# Copyright 2017  Ruhr-University Bochum (Author: Hendrik Meutzner)
#
# Apache 2.0.

# This script summarizes all results that are found in the exp-dir.
# The scores can also be loaded from multiple exp directories via --exp or multiple text files containing collected scores scores via --file.
# The table output can be limited to a customized list of models names via --models.

import os, sys, getopt
import numpy as np

# read keyword recognition accuracies from single file
def read_snr_keyword_recognition_accuracies_single(filename):

    with open(filename, 'r') as fin:

        # read all lines
        data = fin.readlines()

        # get row containing SNR values and remove first and last column
        headers = [d.strip() for d in data[2].split()[1:-1]]

        # get the corresponding keyword recognition accuracies for each SNR
        kwacc = [d.strip() for d in data[4].split()[1:-1]]

        return zip(headers, kwacc)


def collect_scores_from_exp_root_folder(exp_root, scores_filename='keyword_scores.txt', verb=True):

    scores_list = []
    exp_id_list = []

    if verb:
      print "Searching for scores in %s" %(exp_root)

    for root, subFolders, files in os.walk(exp_root):
            
        if scores_filename in files:
        
            if verb:
              print "Reading scores from %s" %(os.path.join(root, scores_filename))
              
            scores = read_snr_keyword_recognition_accuracies_single(os.path.join(root, scores_filename))

            # extract tuple consisting of (feature type, model name, set name + smbr it)
            setup_desc = root[len(exp_root)+1:].split('/')

            # extract last field containing decode_*
            # keep devel/test and possibly the iteration of smbr
            tmp_set_smbr = setup_desc[-1].split('_')[1:]

            # set name, feature type, model name, 
            exp_id = [tmp_set_smbr[0], setup_desc[0], setup_desc[1]]
            if len(tmp_set_smbr) > 1:
                exp_id.append(tmp_set_smbr[1])

            scores_list.append(tuple(scores))
            exp_id_list.append(tuple(exp_id))

    if verb and scores_list:
        print "%d results have been read" %(len(scores_list))

    return exp_id_list, scores_list


# read keyword recognition accuracies from a collection of result files
def collect_scores_from_combined_file(filename, verb=True):

    scores_list = []
    exp_id_list = []

    if verb:
      print "Reading scores from %s" %(filename)
      
    with open(filename, 'r') as fin:
        # read all lines
        data = fin.readlines()

        i = 0
        while i < len(data):
            # check if we have path
            row = data[i]

            if '/' in row:        
                setup_desc = row.split('/')[1:-1]

                headers = [d.strip() for d in data[i+3].split()[1:-1]]
                kwacc = [d.strip() for d in data[i+5].split()[1:-1]]

                scores = zip(headers,kwacc)

                # extract last field containing decode_*
                # keep devel/test and possibly the iteration of smbr
                tmp_set_smbr = setup_desc[-1].split('_')[1:]

                # set name, feature type, model name, 
                exp_id = [tmp_set_smbr[0], setup_desc[0], setup_desc[1]]

                # add data set name used for decoding
                if len(tmp_set_smbr) > 1:
                  exp_id.append(tmp_set_smbr[1])

                scores_list.append(tuple(scores))
                exp_id_list.append(tuple(exp_id))

                i+=9 # skip some rows
            else:
                i+=1

    return exp_id_list, scores_list

def update_score_list(slist, snew):

    if slist:    
        s0, s1 = slist[0], slist[1]
        s0 += snew[0]
        s1 += snew[1]
        scores = [s0, s1]
    else:
        scores = snew
        
    return scores
    
def usage():
    print "Summarize scores.\n"
    print "Scores can be loaded from multiple exp directories via --exp or multiple text files containing collected scores scores via --file."
    print "The output can be limited to a customized list of models names via --models."
    print 'Usage: '+sys.argv[0]+' [--exp <exp root dir> | --file <results file>] [--models <model names>]\n'
    print 'Example 1 (exp dir): '+sys.argv[0]+' --exp "~/kaldi/exp"'
    print 'Example 2 (single file): '+sys.argv[0]+' --file "~/kaldi/results/all.txt"'
    print 'Example 3 (single file and custom models): '+sys.argv[0]+' --file "~/kaldi/results/all.txt" --models "tri3b,dnn"'
    print 'Example 4 (multiple exp dirs): '+sys.argv[0]+' --exp "~/kaldi/exp1,~/kaldi/exp2"'
    print 'Example 5 (multiple files): '+sys.argv[0]+' --file "~/kaldi/results/res1.txt,~/kaldi/results/res2.txt"' 
    print 'Example 6 (all)'+sys.argv[0]+' --exp "~/kaldi/exp1,~/kaldi/exp2" --file "~/kaldi/results/res1.txt,~/kaldi/results/res2.txt,~/kaldi/results/res3.txt --models "tri3b,dnn"'
  
def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hefm",["help","exp=","file=","models="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
    expdirs = []
    single_files = []
    models = []
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-e", "--exp"):
            expdirs = arg
        elif opt in ("-f", "--file"):
            single_files = arg    
        elif opt in ("-m", "--models"):
            models = arg
        
    if max(len(expdirs), len(single_files)) == 0:
        print "No input source specified!"
        usage()
        sys.exit(2)
    
    if expdirs:
        expdirs = expdirs.split(',')
        if not type(expdirs) is list:
            expdirs = list(expdirs)
    
    if single_files:
        single_files = single_files.split(',')
        if not type(single_files) is list:
            single_files = list(single_files)    
  
    if models:
        models = models.split(',')
        if not type(models) is list:
            models = list(models)  
  
    scores = []
    
    if expdirs:    
        for exp in expdirs:            
            tmp = collect_scores_from_exp_root_folder(exp, verb=False)            
            scores = update_score_list(scores, tmp)
    
    if single_files:
        for sf in single_files:            
            tmp = collect_scores_from_combined_file(sf, verb=False)
            scores = update_score_list(scores, tmp)

    scores_sorted = sorted(zip(scores[0], scores[1]), key=lambda x: (x[0][0], x[0][1], x[0][2]))

    tmpStrings = []
    uniqueStrings = []
    for s in scores_sorted:
    
        # get model name
        #model = s[0][2].split('_')[0]
	tmp = s[0][2].split('_')
	model = tmp[0]


	# append version if available	
	if len(tmp) > 1:
		if tmp[1][0] == "v":
	   		model += "-" + tmp[1]
        
        # append smbr iteration
        if len(s[0]) == 4:
            model = "%s (%s)" %(model, s[0][3])
            
        # filter models, if specified
        if models:
            if not model in models:
                continue
                
        # generate one string containing all scores for each snr, e.g.,
        # 68.03   73.89   81.29   86.99   90.99   92.94        
        score_str = '\t'.join([(v[1]) for v in s[1]])
        
        av_scores = np.mean([float(v[1]) for v in s[1]])
        
        # set name \t features \t model \t snr list \t average score
        printStr = "%s\t%-5s\t%-12s\t%s\t| %.2f\n" %(s[0][0], s[0][1], model, score_str, av_scores)
        tmpStrings.append(printStr)
    
    # print header
    headerStr = "%5s\t%-5s\t%-12s\t%-ddB\t%-ddB\t%-ddB\t%-ddB\t%-ddB\t%-ddB\t| %s" %("Set","Feat.","Model",-6,-3,0,3,6,9,"Avg.")
    print "#"*87 + "\n" + headerStr + "\n" + "#"*87

    # remove duplicates and print
    [uniqueStrings.append(row) for row in tmpStrings if row not in uniqueStrings]
    print ''.join(uniqueStrings)

if __name__ == "__main__":
    main(sys.argv[1:])
