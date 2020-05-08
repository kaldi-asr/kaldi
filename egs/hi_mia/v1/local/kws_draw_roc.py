#!/usr/bin/env python

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University(Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation(Authors:Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0

# Draw a roc curve from result file

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readFile(filename):
    f=open(filename,"r")
    datadict=dict()
    for line in f.readlines():
        datadict[line.split()[0]]=line.split()[1]
    return datadict
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="draw roc")
    parser.add_argument('result_file',help='output of network')
    parser.add_argument('label_file',help='utt label')

    FLAGS = parser.parse_args()
    
    resultDic = readFile(FLAGS.result_file)
    labelDic = readFile(FLAGS.label_file)

    x = list()
    y = list()
    for i in range(1,1001):
        false_reject=0
        false_alarm=0
        positive=0
        negative=0
        true_alarm=0
        true_reject=0
        print "shreld"+str(float(i)/1000)
        for key in resultDic:
            label = int(labelDic[key])
            result = float(resultDic[key])
            if label ==1:
                positive=positive+1
                if result<float(i)/1000:
                    false_reject=false_reject+1
                else:
                    true_alarm=true_alarm+1
            elif label ==0:
                negative=negative+1
                if result>float(i)/1000:
                    false_alarm=false_alarm+1
                else:
                    true_reject=true_reject+1
        false_alarm_rate=float(false_alarm)/negative
        false_reject_rate=float(false_reject)/positive
        print float(true_reject+true_alarm)/float(positive+negative)
        x.append(false_alarm_rate)
        y.append(false_reject_rate)
    plt.plot(x,y,linewidth=4)
    plt.xlim(0,0.02)
    plt.ylim(0,1)
    plt.title("roc")
    plt.xlabel("false alarm")
    plt.ylabel("false reject")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
            
