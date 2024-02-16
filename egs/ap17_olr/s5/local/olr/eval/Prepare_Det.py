#!/usr/bin/env python

import sys

langnum = 10

dictl = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10}

# Load scoring file and label.scp.
def Loaddata(fin):
	x = []
	for i in range(langnum+1):
		x.append(0)
	fin = open(fin, 'r')
	lines = fin.readlines()
	fin.close()


	data = []

	for line in lines[1:]:
		part = line.split()
		x[0] = part[0].split('g')[1].split('_')[0]
		for i in range(langnum):
			x[i+1] = part[i + 1]
		data.append(x)
		x = []
		for i in range(langnum+1):
			x.append(0)
	
	return data


# Generate target trials and nontarget trials.
# Prepare for plotting DET curves and computing EER / minDCF.
# data: matrix for result scores.
def fun(data, targetf, nontargetf):
	
	targetf = open(targetf, 'w')
	nontargetf = open(nontargetf, 'w')
	for part in data:
		lan = part[0]
		for j in range(langnum):
			if j + 1 == dictl[lan]:
				targetf.write(part[j + 1] + '\n')
			else:
				nontargetf.write(part[j + 1] + '\n')
	targetf.close()
	nontargetf.close()


if __name__ == '__main__':
    '''
    if (len(sys.argv) != 3):
        print "usage %s <result file path> <label file path>" % (sys.argv[0])
        exit(0)
    '''
    data = Loaddata(sys.argv[1])
    fun(data,'lid_score/DET/target.txt','lid_score/DET/nontarget.txt')
