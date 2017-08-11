#!/usr/bin/env python
# Copyright 2016  Tsinghua University
#                 (Author: Yixiang Chen, Lantian Li, Dong Wang)
# Licence: Apache 2.0


import sys
from math import *


# Load the result file in the following format
# <lang-id> ct-cn     id-id     ja-jf     ko-kr     ru-ru     vi-vn     zh-cn ...
# <utt-id>  <score1>  <score2>  <score3>  <score4>  <score5>  <score6>  <scorelangnum> ...

langnum = 10
dictl = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10}

# Load scoring file and label.scp.
def Loaddata(fin, langnum, lanlabel):
    lanlabel = open(lanlabel, 'r')
    linesll = lanlabel.readlines()
    lanlabel.close()
    
    uttllist = []
    for line in linesll:
        part = line.split()
        uttllist.append(part[0])
    fin = open(fin, 'r')
    lines = fin.readlines()
    fin.close()

    datas = []
    i=0
    for line in lines:
        part = line.split()
        datas.append([dictl[uttllist[i]], 1, float(part[1])])
        datas.append([dictl[uttllist[i]], 2, float(part[2])])
        datas.append([dictl[uttllist[i]], 3, float(part[3])])
        datas.append([dictl[uttllist[i]], 4, float(part[4])])
        datas.append([dictl[uttllist[i]], 5, float(part[5])])
        datas.append([dictl[uttllist[i]], 6, float(part[6])])
        datas.append([dictl[uttllist[i]], 7, float(part[7])])
	datas.append([dictl[uttllist[i]], 8, float(part[8])])
	datas.append([dictl[uttllist[i]], 9, float(part[9])])
	datas.append([dictl[uttllist[i]], 10, float(part[10])])
        i += 1
    print datas[0]
    # score normalized to [0, 1] 
    for i in range(len(datas) / langnum):
        sum = 0
        for j in range(langnum):
            k = i * langnum + j
            sum += exp(datas[k][2])
        for j in range(langnum):
            k = i * langnum + j
            datas[k][2] = exp(datas[k][2]) / sum


    return datas

# Compute Cavg.
# data: matrix for result scores, assumed within [0,1].
# sn: number of bins in Cavg calculation.
def CountCavg(data, sn = 20, lgn = 4):

    Cavg = [0.0] * (sn + 1) 
    # Cavg: Every element is the Cavg of the corresponding precision
    precision = 1.0 / sn
    for section in range(sn + 1):
        threshold = section * precision
        target_Cavg = [0.0] * lgn
        # target_Cavg: P_Target * P_Miss + sum(P_NonTarget*P_FA)

        for language in range(lgn):
            P_FA = [0.0] * lgn
            P_Miss = 0.0
            # compute P_FA and P_Miss
            LTm = 0.0; LTs = 0.0; LNm = 0.0; LNs = [0.0] * lgn;
            for line in data:
                language_label = language + 1
                if line[0] == language_label:
                    if line[1] == language_label:
                        LTm += 1
                        if line[2] < threshold:
                            LTs += 1
                    for t in range(lgn):
                        if not t == language:
                            if line[1] == t + 1:
                                if line[2] > threshold:
                                    LNs[t] += 1
            LNm = LTm
            for Ln in range(lgn):
                P_FA[Ln] = LNs[Ln] / LNm
                
            P_Miss = LTs / LTm
            P_NonTarget = 0.5 / (lgn - 1)
            P_Target = 0.5
            target_Cavg[language] = P_Target * P_Miss + P_NonTarget*sum(P_FA)

        for language in range(lgn):
            Cavg[section] += target_Cavg[language] / lgn   
            
    return Cavg, min(Cavg)


if __name__ == '__main__':

    fin=sys.argv[1]
    lanlabel = sys.argv[2]
    data = Loaddata(fin, langnum, lanlabel)
    
    # default precision as 20 bins, langnum languages
    cavg, mincavg = CountCavg(data, 20, langnum)

    print "Cavg is :" + str(cavg)
    print "Minimal Cavg is :" + str(mincavg) +'\n'



