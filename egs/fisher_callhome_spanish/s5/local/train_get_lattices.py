#!/usr/bin/env python
# Copyright 2014  Gaurav Kumar.   Apache 2.0

from __future__ import print_function
import os
import sys
import subprocess

latticeLocation = {1:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-1/latjosh-2/lattices-pushed/",
2:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-2/latjosh-2/lattices-pushed/",
3:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-3/latjosh-2/lattices-pushed/",
4:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-4/latjosh-2/lattices-pushed/",
5:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-5/latjosh-2/lattices-pushed/",
6:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-6/latjosh-2/lattices-pushed/",
7:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-7/latjosh-2/lattices-pushed/",
8:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-8/latjosh-2/lattices-pushed/",
9:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-9/latjosh-2/lattices-pushed/",
10:"/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-10/latjosh-2/lattices-pushed/"}

latticeDict = {}

for key,location in latticeLocation.items():
    for root, dirs, filenames in os.walk(location):
        for f in filenames:
            latticeDict[f] = str(key)

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
        if os.path.exists('rm ' + tmpdir + '/tmp.lat'):
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
    searchKey = timeDetail + '.lat'
    if searchKey in latticeDict:
        return "/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-" + latticeDict[searchKey] + "/latjosh-2/lattices-pushed/" + searchKey
    else:
        return -1


# Now read list of files in conversations
fileList = []
conversationList = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/train')
for line in conversationList:
    line = line.strip()
    line = line[:-4]
    fileList.append(line)

# IN what order were the conversations added to the spanish files?
# Now get timing information to concatenate the ASR outputs

provFile = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/asr.train.plf', 'w+')
lineNo = 1
invalidPLF = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/invalidPLF', 'w+')
blankPLF = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/blankPLF', 'w+')
rmLines = open('/export/a04/gkumar/corpora/fishcall/jack-splits/split-matt/removeLines', 'w+')
for item in fileList:
    timingFile = open('/export/a04/gkumar/corpora/fishcall/fisher/tim/' + item + '.es')
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

        if mergedTranslation != "":
            
            # Sanjeev's Recipe : Remove epsilons and topo sort
            finalFST = tmpdir + "/final.fst"
            os.system("fstrmepsilon " + mergedTranslation + " | fsttopsort - " + finalFST)
        
            # Now convert to PLF
            proc = subprocess.Popen('/export/a04/gkumar/corpora/fishcall/bin/fsm2plf.sh /export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-matt/data/lang/words.clean.txt ' + finalFST, stdout=subprocess.PIPE, shell=True)
            PLFline = proc.stdout.readline()
            finalPLFFile = tmpdir + "/final.plf"
            finalPLF = open(finalPLFFile, "w+")
            finalPLF.write(PLFline)
            finalPLF.close()

            # now check if this is a valid PLF, if not write it's ID in a 
            # file so it can be checked later
            proc = subprocess.Popen("/export/a04/gkumar/moses/mosesdecoder/checkplf < " + finalPLFFile + " 2>&1 | awk 'FNR == 2 {print}'", stdout=subprocess.PIPE, shell=True)
            line = proc.stdout.readline()
            print("{} {}".format(line, lineNo))
            if line.strip() != "PLF format appears to be correct.":
                os.system("cp " + finalFST + " " + invalidplfdir + "/" + timeInfo[0])
                invalidPLF.write(invalidplfdir + "/" + timeInfo[0] + "\n")
                rmLines.write("{}\n".format(lineNo))
            else:
                provFile.write(PLFline)
        else:
            blankPLF.write(timeInfo[0] + "\n")
            rmLines.write("{}\n".format(lineNo))
        # Now convert to PLF
        lineNo += 1

provFile.close()
invalidPLF.close()
blankPLF.close()
rmLines.close()
