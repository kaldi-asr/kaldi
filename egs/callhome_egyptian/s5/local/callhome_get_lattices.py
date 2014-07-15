#!/usr/bin/env python

# Extracts one best output for a set of files
# What do you need?
# Location of the timing files
# The list of files in the conversations for which 1 best output has to be extracted
# words.txt 

import os
import sys
import subprocess

latticeLocation = 'latjosh-2-callhome/lattices-pushed/'
tmpdir = 'data/local/data/tmp/lattmp'
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)
invalidplfdir = 'data/local/data/tmp/invalidplf'
if not os.path.exists(invalidplfdir):
    os.makedirs(invalidplfdir)
else:
    os.system("rm " + invalidplfdir + "/*")

def latticeConcatenate(lat1, lat2):
    '''
    Concatenates lattices, writes temporary results to tmpdir
    '''
    if lat1 == "":
        os.system('rm ' + tmpdir + '/tmp.lat')
        return lat2
    else:
        proc = subprocess.Popen(['fstconcat', lat1, lat2, (tmpdir + '/tmp.lat')])
        proc.wait()
        return tmpdir + '/tmp.lat'


def findLattice(timeDetail):
    '''
    Finds the lattice corresponding to a time segment
    ''' 
    if os.path.isfile(latticeLocation + timeDetail + '.lat'):
        return latticeLocation + timeDetail + '.lat'
    else:
        return -1


# Now read list of files in conversations
fileList = []
conversationList = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/train')
for line in conversationList:
    line = line.strip()
    line = line[:-4]
    fileList.append(line)

# IN what order were the conversations added to the spanish files?
# Now get timing information to concatenate the ASR outputs

provFile = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/asr.train.plf', 'w+')
lineNo = 1
invalidPLF = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/invalidPLF', 'w+')
blankPLF = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/blankPLF', 'w+')
rmLines = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-callhome/removeLines', 'w+')
for item in fileList:
    timingFile = open('/export/a04/gkumar/corpora/fishcall/callhome/tim/' + item + '.es')
    for line in timingFile:
        timeInfo = line.split()

        # For utterances that are concatenated in the translation file, 
        # the corresponding FSTs have to be translated as well
        mergedTranslation = ""
        for timeDetail in timeInfo:
            tmp = findLattice(timeDetail)
            if tmp != -1:
                # Concatenate lattices
                mergedTranslation = latticeConcatenate(mergedTranslation, tmp)

        print mergedTranslation
        if mergedTranslation != "":
            
            # Sanjeev's Recipe : Remove epsilons and topo sort
            finalFST = tmpdir + "/final.fst"
            os.system("fstrmepsilon " + mergedTranslation + " | fsttopsort - " + finalFST)
        
            # Now convert to PLF
            proc = subprocess.Popen('/export/a04/gkumar/corpora/fishcall/bin/fsm2plf.sh /export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-matt/exp/tri5a/graph/words.txt ' + finalFST, stdout=subprocess.PIPE, shell=True)
            PLFline = proc.stdout.readline()
            finalPLFFile = tmpdir + "/final.plf"
            finalPLF = open(finalPLFFile, "w+")
            finalPLF.write(PLFline)
            finalPLF.close()

            # now check if this is a valid PLF, if not write it's ID in a 
            # file so it can be checked later
            proc = subprocess.Popen("/export/a04/gkumar/moses/mosesdecoder/checkplf < " + finalPLFFile + " 2>&1 | awk 'FNR == 2 {print}'", stdout=subprocess.PIPE, shell=True)
            line = proc.stdout.readline()
            print line + " " + str(lineNo)
            if line.strip() != "PLF format appears to be correct.":
                os.system("cp " + finalFST + " " + invalidplfdir + "/" + timeInfo[0])
                invalidPLF.write(invalidplfdir + "/" + timeInfo[0] + "\n")
                rmLines.write(str(lineNo) + "\n")
            else:
                provFile.write(PLFline)
        else:
            blankPLF.write(timeInfo[0] + "\n")
            rmLines.write(str(lineNo) + "\n")
        # Now convert to PLF
        lineNo += 1

provFile.close()
invalidPLF.close()
blankPLF.close()
rmLines.close()
