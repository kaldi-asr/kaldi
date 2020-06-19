#!/usr/bin/env python3
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0
#
# This script is for keyword filler system to do fst

import numpy as np
import argparse
import sys
import copy
class Edge:
    def __init__(self, cur_state, to_state, isymbol, osymbol):
        self.cur_state = int(cur_state)
        self.to_state = int(to_state)
        self.isymbol = int(isymbol)
        self.osymbol = int(osymbol)
    
    def getToState(self):
        return self.to_state

class State:
    def __init__(self):
        self.oedge = list()
        self.inputSymbol = list()
    def addEdge(self,edge):
        self.oedge.append(edge)
        self.inputSymbol.append(int(edge.isymbol))
    def hasOut(self,inputId):
        for i in self.inputSymbol:
            if i == inputId:
                return True
        return False
    
    def getOutput(self, inputId):
        if inputId in self.inputSymbol:
            pos = self.inputSymbol.index(inputId)
        else:
            return -1,-1
        if pos == -1:
            sys.exit(1)
        return self.oedge[pos].to_state, self.oedge[pos].osymbol

class Token:
    def __init__(self):
        self.active = False
        self.is_filler = True
        self.score = 0
        self.num_keyword_frames = 0
        self.average_keyword_score = 0.0
        self.keyword = 0
        self.num_frames_of_current_state = 0
        self.num_keyword_states = 0
        self.max_score_of_current_state = 0.0
        self.average_max_keyword_score = 0.0
        self.average_max_keyword_score_before = 0.0
    def transform(self,prev,olabel,is_self_arc,is_filler,score):
        if(self.active == False or (self.active == True and self.score < prev.score + score)):
            self.score = prev.score + score
            if(is_filler == False):
                t = prev.num_keyword_frames
                average_keyword_score = (score + prev.average_keyword_score * t) / (t + 1); 
                num_keyword_frames = t + 1
                if is_self_arc:
                    self.num_frames_of_current_state = prev.num_frames_of_current_state + 1
                    self.num_keyword_states = prev.num_keyword_states
                    self.max_score_of_current_state = max(prev.max_score_of_current_state, score)
                    self.average_max_keyword_score_before = prev.average_max_keyword_score_before
                else:
                    self.num_frames_of_current_state = 1
                    self.num_keyword_states = prev.num_keyword_states + 1
                    self.max_score_of_current_state = score
                    self.average_max_keyword_score_before = prev.average_max_keyword_score
                self.average_keyword_score=(self.max_score_of_current_state + 
                    self.average_max_keyword_score_before * (self.num_keyword_states - 1)) / self.num_keyword_states
                if olabel != 0:
                    self.keyword = olabel
            self.active = True
            self.is_filler = is_filler
                
    def reset():
        self.active = False
        self.is_filler = True
        self.score = 0
        self.num_keyword_frames = 0
        self.average_keyword_score = 0.0
        self.keyword = 0
        self.num_frames_of_current_state = 0
        self.num_keyword_states = 0
        self.max_score_of_current_state = 0.0
        self.average_max_keyword_score = 0.0
        self.average_max_keyword_score_before = 0.0


class StateMachine:
    def __init__(self, stateFile):
        self.stateList = list()
        f = open(stateFile,"r")
        self.endState = dict()
        self.endState_id = 0
        for line in f.readlines():
            if len(line.split()) == 2:
                self.endState[line.split()[0]] = float(line.split()[1])
                self.endState_id = int(line.split()[0])
                continue
            if  len(self.stateList) <= int(line.split()[0]):
                new_state = State()
                new_state.addEdge(Edge(line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
                
                self.stateList.append(new_state)
            else:
                self.stateList[int(line.split()[0])].addEdge(Edge(line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        f.close
        self.name = ""
        self.cur_tokens = [Token() for i in range(len(self.stateList))]
        self.prev_tokens = [ Token() for i in range(len(self.stateList))]
        self.beamsize = 10
        self.frame_num = 0
        self.keyword_id = 0
        self.spot = False
    
    def runFst(self, postFile):
        f = open(postFile, "r")
        self.postList = dict()
        i=0
        for line in f.readlines():
            self.runFst_oneLine(line)
            i = i+1
        return self.postList

    def runFst_oneLine(self,line):
        if(len(line.split()) == 2):
            self.name = line.split()[0]
            self.frame_num = 0
            self.cur_tokens = [Token() for i in range(len(self.stateList))]
            self.prev_tokens = [ Token() for i in range(len(self.stateList))]
            self.prev_tokens[0].active=True
            self.keyword_id = 0
            self.spot = False

        elif(len(line.split()) == 10):
            self.frame_num = self.frame_num + 1
            post = [ np.exp(float(x)) for x in line.split() ]
            post= np.log(post/np.sum(post))
                    
            for num,tokens in enumerate(self.prev_tokens):
                if tokens.active:
                    for i in self.stateList[num].oedge:  
                        score = post[i.isymbol]
                        is_filler = (i.isymbol <= 2)
                        is_self_arc=(num==i.getToState())
                        self.cur_tokens[i.getToState()].transform(tokens,i.osymbol,is_self_arc,is_filler,score)
            best_state = 0
            best_final_state = 0
            best_score = self.cur_tokens[0].score
            best_final_score = 0.0
            reach_final = False
            for i in range(1,len(self.cur_tokens)):
                if self.cur_tokens[i].active and best_score < self.cur_tokens[i].score:
                    best_score = self.cur_tokens[i].score
                    best_state = i
                if self.cur_tokens[i].active and i == self.endState_id:
                    if(reach_final == False):
                        best_final_state = i
                        best_final_score = self.cur_tokens[i].score
                        reach_final=True
                    elif best_final_score < self.cur_tokens[i].score:
                        best_final_state = i
                        best_final_score = self.cur_tokens[i].score
            # print best_state
            # print best_score
            if reach_final:
                self.postList[self.name] = np.exp(self.cur_tokens[best_final_state].average_keyword_score)
                self.keyword_id = self.cur_tokens[best_final_state].keyword
                if self.cur_tokens[best_final_state].num_keyword_frames >= 0 and \
                    self.cur_tokens[best_final_state].num_frames_of_current_state >= 5 and \
                    self.postList[self.name] > 0.5:
                    self.spot = True
            self.prev_tokens = self.cur_tokens
            self.cur_tokens = [ Token() for i in range(len(self.stateList))]
            self.frame_num = self.frame_num + 1
            if self.frame_num > 100*60*10 and self.prev_tokens[best_state].is_filler:
                self.prev_tokens=[ Token() for i in range(len(self.stateList))]

        elif(len(line.split()) == 11):
            if self.spot == False:
                self.postList[self.name] = np.exp(self.prev_tokens[self.endState_id].average_keyword_score)
            
    
    def printFst(self):
        for i in range(len(self.stateList)):
            print("state" + str(i))
            print(self.stateList[i].inputSymbol)


    
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="This script is for keyword filler system to do fst\n")
    parser.add_argument('Fst_file',help='Fst file')
    parser.add_argument('bnf_file',help='output of network')
    
    FLAGS = parser.parse_args()

    stateM = StateMachine(FLAGS.Fst_file)
    result = stateM.runFst(FLAGS.bnf_file)

    for k,v in result.items():
        print(str(k) + " " + str(v))
