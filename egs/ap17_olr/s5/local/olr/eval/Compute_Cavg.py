#!/usr/bin/env python
# Copyright 2016  Tsinghua University
#                 (Author: Yixiang Chen, Lantian Li, Dong Wang)
# Licence: Apache 2.0


import sys
from math import *


# Load the result file in the following format
#           lang0     lang1     lang2     lang3     lang4     lang5     lang6     lang7     lang8     lang9   
# <utt-id>  <score0>  <score1>  <score2>  <score3>  <score4>  <score5>  <score6>  <score7>  <score8>  <score9>

# The language identity is defined as: 
# {'ct-cn':'0', 'id-id':'1', 'ja-jp':'2', 'ko-kr':'3', 'ru-ru':'4', 'vi-vn':'5', 'zh-cn':'6', 'Kazak':'7', 'Tibet':'8', 'Uyghu':'9'}

langnum = 10
dictl = {'0':1, '1':2, '2':3, '3':4, '4':5, '5':6, '6':7, '7':8, '8':9, '9':10}

# Load scoring file and label.scp.
def Loaddata(fin, langnum):

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

    datas = []
    for ll in data:
        for lb in range(langnum):
            datas.append([dictl[ll[0][0]], lb + 1, float(ll[lb + 1])])
            

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
    data = Loaddata(fin, langnum)
    
    # default precision as 20 bins, langnum languages
    cavg, mincavg = CountCavg(data, 20, langnum)

    #print "Cavg is :" + str(cavg)
    print "Minimal Cavg is: " + str(round(mincavg,4)) +' '

