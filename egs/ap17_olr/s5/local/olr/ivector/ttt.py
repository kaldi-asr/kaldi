#!/usr/bin/env python

import sys

def fun(fin, fout):
	fin = open(fin, 'r')
	linef = fin.readlines()
	fin.close()
	
	fout = open(fout, 'w')
	for line in linef:
		part = line.split()
		if part[0]=='[[':
			feats = part
		else:
			feats.extend(part)
		if len(feats) == 11:
			out = ''
                	for pp in feats:
                        	out = out + pp +' '
                	out = out + '\n'
                	fout.write(out)
	fout.close()

if __name__ == '__main__':
	fin = sys.argv[1]
	fout = sys.argv[2]
	
	fun(fin, fout)
