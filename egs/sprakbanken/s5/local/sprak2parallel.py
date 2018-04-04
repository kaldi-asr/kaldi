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
'''

import subprocess                                                                                                     
import sys                                                                                                            
import codecs                                                                                                         
import os                                                                                                             
from sprakparser import Session                                                                                       
import shutil

n = 0

### Utility functions                                                                                                 
def find_ext_folders(topfolder, extfolderlist, file_ext):                                                             
    '''Recursive function that finds all the folders containing $file_ext files and                                   
    returns a list of folders.'''                                                                                     
                                                                                                                      
    for path in os.listdir(topfolder):
        curpath = os.path.join(topfolder,path)                                                                        
        if os.path.isdir(curpath):                                                                                    
            find_ext_folders(curpath, extfolderlist, file_ext)                                                        
        elif os.path.isfile(curpath):                                                                                 
            if os.path.splitext(path)[1] == file_ext:                                                                 
                extfolderlist.append(topfolder)                                                                       
                return                                                                                                
        else:         
            pass 


def create_parallel_files(session):
    
    for recnum, recording in enumerate(session.record_states):
        if recnum == 0: # skip the first recording of silence
            continue
        oldsound = os.path.join(session.wavdir, recording[1])
        if not os.path.exists(oldsound):
            continue
        txtout = session.create_filename(recnum+1, "txt")
        sndout = session.create_filename(recnum+1, "wav")
        
        fout = codecs.open(os.path.join(session.sessiondir,txtout), "w", "utf8")
        fout.write(recording[0]+ "\n")
        fout.close()
        dst = shutil.copyfile(oldsound, os.path.join(session.sessiondir,sndout))
        

def make_speech_corpus(top, dest, srcfolder):
    global n
    spls = os.listdir(srcfolder)
    for splfile in sorted(spls):
        if os.path.splitext(splfile)[1] != ".spl":
            continue
        
        session = Session(os.path.abspath(srcfolder), splfile)
        if session.speaker_id == "": # ignore if there is no speaker
            continue
        if not session.wavdir: # ignore if there is no matching directory
            continue
        if len(session.record_states) < 2: # unsure whether this has an effect
            continue
        session.sessiondir = os.path.join(dest, session.filestem) +"."+ session.speaker_id
        if os.path.exists(session.sessiondir):
            n += 1
            session.sessiondir = session.sessiondir+ "_" +str(n)
            session.speaker_id+ "_" +str(n)
        os.mkdir(session.sessiondir)
        
        create_parallel_files(session)


if __name__ == '__main__':

    spldirs = []
    dest = sys.argv[2] #"/mnt/sprakkaldi"
    if not os.path.exists(dest):
        os.mkdir(dest)
    topfolder = sys.argv[1]
    find_ext_folders(topfolder, spldirs, ".spl")
    
#    print(len(spldirs))
    for num, folder in enumerate(spldirs):
        make_speech_corpus(topfolder, dest, folder)
