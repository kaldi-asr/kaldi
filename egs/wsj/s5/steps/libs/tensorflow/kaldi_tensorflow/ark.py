##@package ark 
#contains the .ark io functionality

# Copyright 2014    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

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

import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)
import struct

	
## Class to read Kaldi ark format. Each time, it reads one line of the .scp file and reads in the corresponding features into a numpy matrix. It only supports binary-formatted .ark files. Text and compressed .ark files are not supported. The inspiration for this class came from pdnn toolkit (see licence at the top of this file)(https://github.com/yajiemiao/pdnn)
class ArkReader(object):
	
	##ArkReader constructor
	#@param scp_path path to the .scp file
	def __init__(self, scp_path):
	
		self.scp_position = 0
		fin = open(scp_path,"r")
		self.utt_ids = []
		self.scp_data = []
		line = fin.readline()
		while line != '' and line != None:
			utt_id, path_pos = line.replace('\n','').split(' ')
			path, pos = path_pos.split(':')
			self.utt_ids.append(utt_id)
			self.scp_data.append((path, pos))
			line = fin.readline()

		fin.close()
		
	## read data from the archive
	#@param index index of the utterance that will be read
	#@return a numpy array containing the data from the utterance
	def read_utt_data(self, index):
		ark_read_buffer = open(self.scp_data[index][0], 'rb')
		ark_read_buffer.seek(int(self.scp_data[index][1]),0)
		header = struct.unpack('<xcccc', ark_read_buffer.read(5))
		if header[0] != "B":
			print "Input .ark file is not binary"; exit(1)
		if header[1] == "C":
			print "Input .ark file is compressed"; exit(1)

		rows = 0; cols= 0
		m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
		n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

		if header[1] == "F":
			tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
		elif header[1] == "D":
			tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8), dtype=np.float64)
			
		utt_mat = np.reshape(tmp_mat, (rows, cols))

		ark_read_buffer.close()
		
		return utt_mat
	
	## read the next utterance in the scp file
	#@return the utterance ID of the utterance that was read, the utterance data, bool that is true if the reader looped back to the beginning
	def read_next_utt(self):
		
		if len(self.scp_data) == 0:
			return None , None, True 
		
		if self.scp_position >= len(self.scp_data): #if at end of file loop around
			looped = True
			self.scp_position = 0
		else: 
			looped = False
		
		self.scp_position += 1
		
		return self.utt_ids[self.scp_position-1], self.read_utt_data(self.scp_position-1), looped
		
	## read the next utterance ID but don't read the data
	#@return the utterance ID of the utterance that was read
	def read_next_scp(self):
		
		if self.scp_position >= len(self.scp_data): #if at end of file loop around
			self.scp_position = 0
			
		self.scp_position += 1
		
		return self.utt_ids[self.scp_position-1]
		
	## read the previous utterance ID but don't read the data
	#@return the utterance ID of the utterance that was read
	def read_previous_scp(self):
		
		if self.scp_position < 0: #if at beginning of file loop around
			self.scp_position = len(self.scp_data) - 1
			
		self.scp_position -= 1
		
		return self.utt_ids[self.scp_position+1]
	
	## read the data of a certain utterance ID
	#@return the utterance data corresponding to the ID
	def read_utt(self, utt_id):
		
		return self.read_utt_data(self.utt_ids.index(utt_id))
		
	##Split of the data that was read so far
	def split(self):
		self.scp_data = self.scp_data[self.scp_position:-1]
		self.utt_ids = self.utt_ids[self.scp_position:-1]
	
		
        
## Class to write numpy matrices into Kaldi .ark file and create the corresponding .scp file. It only supports binary-formatted .ark files. Text and compressed .ark files are not supported. The inspiration for this class came from pdnn toolkit (see licence at the top of this file)(https://github.com/yajiemiao/pdnn)
class ArkWriter(object):

	##Arkwriter constructor
	#@param scp_path path to the .scp file that will be written
	def __init__(self, scp_path):

		self.scp_path = scp_path
		self.scp_file_write = open(self.scp_path,"w")
    
    ##read an utterance to the archive
    #@param ark_path path to the .ark file that will be used for writing
    #@param utt_id the utterance ID
    #@param utt_mat a numpy array containing the utterance data
	def write_next_utt(self, ark_path, utt_id, utt_mat):
		ark_file_write = open(ark_path,"ab")
		utt_mat = np.asarray(utt_mat, dtype=np.float32)
		rows, cols = utt_mat.shape
		ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
		pos = ark_file_write.tell()
		ark_file_write.write(struct.pack('<xcccc','B','F','M',' '))
		ark_file_write.write(struct.pack('<bi', 4, rows))
		ark_file_write.write(struct.pack('<bi', 4, cols))
		ark_file_write.write(utt_mat)
		self.scp_file_write.write('%s %s:%s\n' % (utt_id, ark_path, pos))
		ark_file_write.close()
		
	##close the ark writer
	def close(self):
		self.scp_file_write.close()
		
