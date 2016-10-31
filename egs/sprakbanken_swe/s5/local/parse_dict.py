'''
# Copyright 2016 KTH Royal Institute of Technology (Author: Emelie Kullmann)

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

import codecs
import sys
import re

#parses the dictionary down from 50 fields to 2, orthographic and (one or multiple) phonetic transcription
#see NST for information about the structure of the original dictionary


transcript = codecs.open(sys.argv[1], "r", "ISO-8859-1")
outtext = codecs.open(sys.argv[2], "w", "utf8")
splits = [':','I','0','a','U','E','Y','e','9','O','@','p','b','t','`','d','k','g','m','N','n','f','v','s','h','r','l','j','T','D','S',"'"]

for line in transcript:
	post = re.split(';',line)
	trans = post[11]
	trans = trans.replace('x\\','S')
	trans = trans.replace('$',"")
	trans = trans.replace('¤',"")
	trans = trans.replace("_","")
	trans = trans.replace('""','"')

	final = ""
	for itr in range(0,len(trans)-1):
		final = final + trans[itr]
		if trans[itr] in splits:
			if not trans[itr+1]== '`' and not trans[itr+1]==':' and not trans[itr+1]=='*' and not trans[itr+1]=="'":
				final = final + " "
	final = final + trans[len(trans)-1]
	post[0] = post[0].replace("_","")
	outtext.write(post[0].upper() + "	" + final + "\n")

	
transcript.close()
outtext.close()
