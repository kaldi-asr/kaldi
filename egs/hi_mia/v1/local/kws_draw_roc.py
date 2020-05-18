#!/usr/bin/env python3
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0                                                                                                        
#
# draw roc curve or get fa per hour and recall

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readFile(filename):
    f = open(filename,"r")
    datadict = dict()
    for line in f.readlines():
        datadict[line.split()[0]] = line.split()[1]
    return datadict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "draw roc")
    parser.add_argument('result_file',help = 'output of network')
    parser.add_argument('label_file',help = 'utt label')
    parser.add_argument('utt2dur_file',help = 'every utt durition')
    parser.add_argument('--roc', action = 'store_true', default = False, help='True to draw roc, false to get fa per hour and recall')

    FLAGS = parser.parse_args()
    
    resultDic = readFile(FLAGS.result_file)
    labelDic = readFile(FLAGS.label_file)
    duritionDic = readFile(FLAGS.utt2dur_file)

    x = list()
    y = list()
    for i in range(1,101):
        false_reject = 0
        false_alarm = 0
        positive = 0
        positive_dur = 0.0
        negative = 0
        negative_dur = 0.0
        true_alarm = 0
        true_reject = 0
        print("shreld" + str(float(i) / 100))
        for key in resultDic:
            label = int(labelDic[key])
            result = float(resultDic[key])
            durition = float(duritionDic[key])
            if label == 1:
                positive_dur = positive_dur + durition
                positive = positive + 1
                if result < float(i) / 100:
                    false_reject = false_reject + 1
                else:
                    true_alarm = true_alarm + 1
            elif label == 0:
                negative_dur = negative_dur + durition
                negative = negative + 1
                if result > float(i) / 100:
                    false_alarm = false_alarm + 1
                else:
                    true_reject = true_reject + 1
        if FLAGS.roc == True:

            false_alarm_rate=float(false_alarm)/negative
            false_reject_rate=float(false_reject)/positive
            print float(true_reject+true_alarm)/float(positive+negative)

            x.append(false_alarm_rate)
            y.append(false_reject_rate)
        else:
            false_alarm_per_hour = float(false_alarm) / (negative * 60)
            recall = float(true_alarm + 1e-5) / (true_alarm + false_alarm + 1e-5) 
            print("false alarm per hour: "+str(false_alarm_per_hour)+", recall: "+str(recall))
            x.append(false_alarm_per_hour)
            y.append(recall)
    plt.plot(x,y,linewidth = 4)
    plt.title("roc")
    plt.xlabel("false alarm")
    plt.ylabel("false reject")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
