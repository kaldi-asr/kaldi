#!/usr/bin/env python

import sys

def fun(fin, score, fout):
	fin =open(fin, 'r')
	linef = fin.readlines()
	fin.close()

	score = open(score, 'r')
	linesc = score.readlines()
	score.close()

	fout = open(fout, 'w')

	labellist = []
	for line in linef:
		part = line.split()
		labellist.append(part[0])

	for i in range(len(linesc)):
		part = linesc[i].split()
		for j in range(len(part)-1):
			out = part[j+1]
			if j == (int(labellist[i])):
				out = out +' target'
			else:
				out = out +' nontarget'
			out  = out + '\n'
			fout.write(out)

	fout.close()

if __name__ == '__main__':
	fin = sys.argv[1]
	score = sys.argv[2]
	fout = sys.argv[3]
	fun(fin, score, fout)
