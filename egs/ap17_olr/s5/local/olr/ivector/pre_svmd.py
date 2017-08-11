#!/usr/bin/env python

import sys

ldt = {'ct':0, 'id':1, 'ja':2, 'Kazak':3, 'ko':4, 'ru':5, 'Tibet':6, 'Uyghu':7, 'vi':8, 'zh':9}

def fun7(fin, lan, fout):
        fin = open(fin, 'r')
        lines = fin.readlines()
        fin.close()

        lan = open(lan, 'r')
        linel = lan.readlines()
        lan.close()

        landict = {}
        for line in linel:
                part = line.split()
                landict[part[0]] = part[1]

        fout = open(fout, 'w')
        for line in lines:
                pt = line.split()
                if landict[pt[0]] == 'ct':
                        k=0
                elif landict[pt[0]] == 'id':
                        k=1
                elif landict[pt[0]] == 'ja':
                        k=2
                elif landict[pt[0]] == 'Kazak':
                        k=3
                elif landict[pt[0]] == 'ko':
                        k=4
                elif landict[pt[0]] == 'ru':
                        k=5
                elif landict[pt[0]] == 'Tibet':
                        k=6
		elif landict[pt[0]] == 'Uyghu':
                        k=7
		elif landict[pt[0]] == 'vi':
                        k=8
		elif landict[pt[0]] == 'zh':
                        k=9
                out = str(k) +' '
                for pp in pt[2:-1]:
                        out = out + pp +' '
                out = out +'\n'
                fout.write(out)
        fout.close()

if __name__ == '__main__':
	fin = sys.argv[1]
	lan = sys.argv[2]
	fout = sys.argv[3]
	fun7(fin, lan, fout)
