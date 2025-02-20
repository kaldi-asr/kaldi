#!/usr/bin/env python
# Copyright 2016  Tsinghua University
#                 (Author: Yixiang Chen, Lantian Li, Dong Wang)
# Licence: Apache 2.0


import sys
from math import *


# Load the result file in the following format
# <lang-id> ct-cn     id-id     ja-jf     ko-kr     ru-ru     vi-vn     zh-cn ...
# <utt-id>  <score1>  <score2>  <score3>  <score4>  <score5>  <score6>  <score7> ...

langnum = 10
dictl = {'ct':1, 'id':2, 'ja':3, 'Kazak':4, 'ko':5, 'ru':6, 'Tibet':7, 'Uyghu':8, 'vi':9, 'zh':10}

# Load scoring file and label.scp.
def Loaddata(fin, landictf):
	x = []
	for i in range(langnum+1):
		x.append(0)
	fin = open(fin, 'r')
	lines = fin.readlines()
	fin.close()
	
	lanf = open(landictf, 'r')
	linesf = lanf.readlines()
	lanf.close()

	landict={}
	for line in linesf:
		part = line.split()
		landict[part[0]] = part[1]
		

	data = []
	
	uttnum = len(lines)/10
	for i in range(uttnum):
		for j in range(10):
			part = lines[10 * i + j].split()
			x[0] = landict[part[1]]
			x[j+1] = part[2]
		data.append(x)
                x = []
                for i in range(langnum+1):
                        x.append(0)

	return data


# Compute IDR.
# data: matrix for result scores.
def CountIdr(data):

	utt_num = len(data)
	bingo_num = 0
	k = 0
	for part in data:
		score = []
		for sc in part[1:]:
			score.append(float(sc))
		bingo = score[dictl[part[0]] - 1]
		score = sorted(score)
		if score[-1] == bingo:
			bingo_num += 1
	idr = float(bingo_num) / float(utt_num)

	return idr


if __name__ == '__main__':


    data = Loaddata(sys.argv[1], sys.argv[2])
    
    idr = CountIdr(data)
    print "IDR is :" + str(idr)
