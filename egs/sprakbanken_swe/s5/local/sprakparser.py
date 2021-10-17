#!/usr/bin/env python
'''
# Copyright 2013-2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)

# Licensed under the Apache License, Version 2.0 (the "License");                                                    
# you may not use this file except in compliance with the License.                                                  
# You may obtain a copy of the License at                                                                          
#                                                                                                                 
#  http://www.apache.org/licenses/LICENSE-2.0                                                                    
#                                                                                                               
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY                                 
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED                                   
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,                                       
# MERCHANTABLITY OR NON-INFRINGEMENT.                                                                                 
# See the Apache 2 License for the specific language governing permissions and                                       
# limitations under the License.


Created on Jan 18, 2013

@author: andreas


'''

import codecs
import os

class Session(object):
    
    delimit = ">-<"
        
    def __init__(self, topfolder, splfile):
        self.filestem = splfile.split(".")[0]
        self.source = topfolder+ "/" +splfile
        splhandle = codecs.open(self.source, "r", "utf8")
        
        self.set_system_vars(splhandle)
        self.set_channel_vars(splhandle)

        self.speaker_id = None
        self.set_speaker_vars(splhandle)

        self. wavdir = False
        self.set_path_vars(topfolder, splhandle)
        if self.wavdir:
            self.extract_record_states(splhandle)
            self.extract_validation_states(splhandle)

    def extract_record_states(self, handle):
        self.record_states = []
        for input in handle:
            if input.strip() == "":
                return
            line = input.split(">-<")
            if len(line) < 4:
                continue
            self.record_states.append((line[2],line[5].lower()))
            

    def extract_validation_states(self, handle):
        self.validation_states = []
        for input in handle:
            if input.strip() == "":
                return
            line = input.split(">-<")
            if len(line) < 4:
                continue
            text = line[0].split("=")[-1]
            qua = line[3]
            noi = line[4]
            snd = line[5]
            spc = line[6]
            utt = line[7]
            dst = line[8]
            self.validation_states.append((text, qua, noi, snd, spc, utt, dst))
                    
    def set_system_vars(self, handle):
        for input in handle:
            line = input.strip()
            #print("sys")
            if "Coding" in line:
                self.coding = self.get_vars(line)
            elif "Frequency" in line:
                self.frequency = self.get_vars(line)
            elif "Channels" in line:
                self.channels = self.get_vars(line)
            elif line == "":
                return
            else:
                pass
                
    def set_speaker_vars(self, handle):
        for input in handle:
            line = input.strip()
            #print("speaker")
            
            if "Speaker" in line:         
                self.speaker_id = self.get_speaker_vars(line).strip()
            elif "Name" in line:
                self.name = self.get_speaker_vars(line)
            elif "Age" in line:
                self.age = self.get_speaker_vars(line)
            elif "Sex" in line:
                self.gender = self.get_speaker_vars(line)
            elif "Youth" in line:
                self.youth = self.get_speaker_vars(line)
            elif "Dialect" in line:
                self.dialect = self.get_speaker_vars(line)
            elif line == "":
                return 
            else:
                pass

    def print_speaker_info(self, path):
        filename = self.filestem+ ".speaker_info"
        dest = os.path.join(path, filename)
        fout = codecs.open(dest, "w", "utf8")
        fout.write("Original .spl file:\t" +self.source)
        fout.write("\nID:\t" +self.speaker_id)
        fout.write("\nName:\t" +self.name)
        fout.write("\nAge:\t" +self.age)
        fout.write("\nGender:\t" +self.gender)
        fout.write("\nDialect:\t" +self.dialect)
        fout.write("\nOrigin:\t" +self.youth)
        fout.close()

            
    def set_path_vars(self, topfolder, handle):
        for input in handle:
            #print("path")
            
            line = input.strip()
            if "recordings" in line:
                self.recordings = self.get_vars(line)
            elif "Directory" in line:
                self.splpath = self.get_vars(line)[3:].replace("\\", "/") # removes "c:\"
                
                self.wavdir = self.wavpath(topfolder)
            elif line == "":
                return 
            else:
                pass
            
    def set_channel_vars(self, handle):
        for input in handle:
            line = input.strip()
            if line == "":
                return
            else:
                pass
            
    def create_filename(self, uid, file_ending):
        return "{}.{}.{}.{}".format(self.filestem, self.speaker_id, uid, file_ending)
        
    def wavpath(self, topfolder):
        prefix, suffix = topfolder.rsplit('/data/', 1)
        testpath = os.path.join(prefix, 'speech', suffix)
        #testpath = topfolder.replace("data", "speech")
        if os.path.exists(testpath):
            return os.path.join(testpath, self.filestem)
        else:
            testpath = os.path.join(prefix, 'Speech', suffix)
            return os.path.join(testpath, self.filestem)
            
    def get_vars(self, line):
        return line.split("=")[-1]
    
    def get_speaker_vars(self, line) :
        vals = self.get_vars(line).split(self.delimit)
        return vals[-2]


