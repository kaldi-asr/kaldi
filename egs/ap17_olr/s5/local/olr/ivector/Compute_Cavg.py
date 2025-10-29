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
dictl = {'ct':1, 'id':2, 'ja':3, 'Kazak':4, 'ko':5, 'ru':6, 'Tibet':7, 'Uyghu':8, 'vi':9, 'zh':10}

# Load scoring file and label.scp.
def Loaddata(fin, landictfin, langnum):
    fin = open(fin, 'r')
    lines = fin.readlines()
    fin.close()

    lanf = open(landictfin, 'r')
    linesl = lanf.readlines()
    lanf.close()
    landict = {}
    for line in linesl:
        part = line.split()
        landict[part[0]] = part[1]
    datas = []

    for line in lines:
        part = line.split()
        datas.append([ dictl[part[0]], dictl[ landict[part[1]] ], float(part[2]) ])
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
    landictf = sys.argv[2]
    data = Loaddata(fin, landictf, langnum)
    
    # default precision as 20 bins, langnum languages
    cavg, mincavg = CountCavg(data, 20, langnum)

    #print "Cavg is :" + str(cavg)
    print "Minimal Cavg is: " + str(round(mincavg,4)) +'\n'



