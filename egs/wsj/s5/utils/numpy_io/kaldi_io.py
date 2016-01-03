#!/usr/bin/env python
# Copyright 2014  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import os, re, gzip, struct

#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
  """ fd = open_or_fd(file)
   Forwards the opened file descriptor or opens the file (or gzipped file).
   Supports pipes, and input records with ':offset'.
   Ignores 'ark:' prefixes (and similar). 
  """
  offset = None
  try:
    # strip 'ark:' prefix from r{x,w}filename (optional),
    if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
      (prefix,file) = file.split(':',1)
    # separate offset from filename (optional),
    if re.search(':[0-9]+$', file):
      (file,offset) = file.rsplit(':',1)
    # is it gzipped?
    if file.split('.')[-1] == 'gz':
      fd = gzip.open(file, mode)
    # input pipe?
    elif file[-1] == '|':
      fd = os.popen(file[:-1], 'rb')
    # output pipe?
    elif file[0] == '|':
      fd = os.popen(file[1:], 'wb')
    # a normal file...
    else:
      fd = open(file, mode)
  except TypeError: 
    # 'file' is opened file descriptor,
    fd = file
  # Eventually seek to offset,
  if offset != None: fd.seek(int(offset)) 
  return fd

def read_key(fd):
  """ [str] = read_key(fd)
   Read the utterance-key from already opened file.
  """
  str = ''
  while 1:
    char = fd.read(1)
    if char == '' : break
    if char == ' ' : break
    str += char
  str = str.strip()
  if str == '': return None # end of file,
  assert(re.match('^[\.a-zA-Z0-9_-]+$',str) != None) # check format,
  return str


#################################################
# Integer vectors (alignments),

def read_vec_int_ark(file_or_fd):
  """ genrator(key,vec) = read_vec_int_ark(file_or_fd)
   Create generator of (key,vector<int>) tuples getting data from ark file.
   file_or_fd : filename or already opened file-descriptor

   Hint, read ark to hash:
   d = {u:d for u,d in pytel.kaldi_io.read_vec_int_ark(file)}
  """
  fd = open_or_fd(file_or_fd)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_int(fd)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_int(file_or_fd):
  """ [int-vec] = read_vec_int(file_or_fd)
   Read integer vector, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd)
  binary = fd.read(2)
  if binary == '\0B': # binary flag
    assert(fd.read(1) == '\4'); # int-size
    vec_size = struct.unpack('<i', fd.read(4))[0] # vector dim
    ans = np.zeros(vec_size, dtype=int)
    for i in range(vec_size):
      assert(fd.read(1) == '\4'); # int-size
      ans[i] = struct.unpack('<i', fd.read(4))[0] #data
    return ans
  else: # ascii,
    arr = (binary + fd.readline()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=int)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans



#################################################
# Float vectors (confidences),

def read_vec_flt_ark(file_or_fd):
  """ genrator(key,vec) = read_vec_flt_ark(file_or_fd)
   Create generator of (key,vector<float>) tuples getting data from ark file.
   file_or_fd : filename or already opened file-descriptor

   Hint, read ark to hash:
   d = { u:d for u,d in pytel.kaldi_io.read_ali_ark(file)}
  """
  fd = open_or_fd(file_or_fd)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_flt(fd)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_flt(file_or_fd):
  """ [flt-vec] = read_vec_flt(file_or_fd)
   Read float vector from file or file-descriptor, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd)
  binary = fd.read(2)
  if binary == '\0B': # binary flag
    # Data type,
    type = fd.read(3)
    if type == 'FV ': sample_size = 4 # floats
    if type == 'DV ': sample_size = 8 # doubles
    assert(sample_size > 0)
    # Dimension,
    assert(fd.read(1) == '\4'); # int-size
    vec_size = struct.unpack('<i', fd.read(4))[0] # vector dim
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32') 
    elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64') 
    else : raise BadSampleSize
    return ans
  else: # ascii,
    arr = (binary + fd.readline()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=float)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

# Writing,
def write_vec_flt(file_or_fd, v, key='', dtype='float32'):
  """ write_vec_flt(f, v, key='')
  Writes a binary kaldi vector to filename or opened file-descriptor. Vector can be float or double.
  Arguments:
   file_or_fd : filename of opened file descriptor for writing,
   v : matrix we are wrining,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.
  """
  fd = open_or_fd(file_or_fd, mode='wb')
  try:
    if key != '' : fd.write(key+' ') # ark-files have keys (utterance-id),
    fd.write('\0B') # we write binary!
    # Data-type,
    v = np.array(v, dtype=dtype) # cast to float32 by default,
    if v.dtype == 'float32': fd.write('FV ')
    elif v.dtype == 'float64': fd.write('DV ')
    else: raise VectorDataTypeError
    # Dim,
    fd.write('\04')
    fd.write(struct.pack('I',v.shape[0])) # dim
    # Data,
    v.tofile(fd, sep="") # binary
  finally:
    if fd is not file_or_fd : fd.close()



#################################################
# Float/double matrices,

# Reading,
def read_mat_scp(file_or_fd):
  """ generator(key,mat) = read_mat_scp(file_or_fd)
   Returns generator of (key,matrix) tuples, which are read from kaldi scp file.
   file_or_fd : filename or already opened file-descriptor (contains scp)

   Hint, read scp to hash:
   d = dict((u,d) for u,d in pytel.kaldi_io.read_mat_scp(file))
  """
  fd = open_or_fd(file_or_fd)
  try:
    for line in fd:
      (key,rxfile) = line.split(' ')
      mat = read_mat(rxfile)
      yield key, mat
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd):
  """ genrator(key,mat) = read_mat_ark(file_or_fd)
   Returns generator of (key,matrix) tuples, which reads ark file.
   file_or_fd : filename or already opened file-descriptor (contains ark)

   Hint, read scp to hash:
   d = dict((u,d) for u,d in pytel.kaldi_io.read_mat_ark(file))
  """
  fd = open_or_fd(file_or_fd)
  try:
    key = read_key(fd)
    while key:
      mat = read_mat(fd)
      yield key, mat
      key = read_key(fd)
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat(file_or_fd):
  """ [mat] = read_mat(file_or_fd)
   Reads kaldi matrix from file or file-descriptor, ascii or binary.
  """
  fd = open_or_fd(file_or_fd)
  try:
    binary = fd.read(2)
    if binary == '\0B' : 
      mat = _read_mat_binary(fd)
    else:
      assert(binary == ' [')
      mat = _read_mat_ascii(fd)
  finally:
    if fd is not file_or_fd: fd.close()
  return mat

def _read_mat_binary(fd):
  # Data type
  type = fd.read(3)
  if type == 'FM ': sample_size = 4 # floats
  if type == 'DM ': sample_size = 8 # doubles
  assert(sample_size > 0)
  # Dimensions
  fd.read(1)
  rows = struct.unpack('<i', fd.read(4))[0]
  fd.read(1)
  cols = struct.unpack('<i', fd.read(4))[0]
  # Read whole matrix
  buf = fd.read(rows * cols * sample_size)
  if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32') 
  elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64') 
  else : raise BadSampleSize
  mat = np.reshape(vec,(rows,cols))
  return mat

def _read_mat_ascii(fd):
  rows = []
  while 1:
    line = fd.readline()
    if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
    if len(line.strip()) == 0 : continue # skip empty line
    arr = line.strip().split()
    if arr[-1] != ']':
      rows.append(np.array(arr,dtype='float32')) # not last line
    else: 
      rows.append(np.array(arr[:-1],dtype='float32')) # last line
      mat = np.vstack(rows)
      return mat

# Writing,
def write_mat(file_or_fd, m, key='', dtype='float32'):
  """ write_mat(f, m, key='')
  Binary write of kaldi matrix. Matrix can be float or double.
  Arguments:
   file_or_fd : filename of opened file descriptor for writing,
   m : matrix we are wrining,
   key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.
  """
  fd = open_or_fd(file_or_fd, mode='wb')
  try:
    if key != '' : fd.write(key+' ') # ark-files have keys (utterance-id),
    fd.write('\0B') # we write binary!
    # Data-type,
    m = np.array(m, dtype=dtype)
    if m.dtype == 'float32': fd.write('FM ')
    elif m.dtype == 'float64': fd.write('DM ')
    else: raise MatrixDataTypeError
    # Dims,
    fd.write('\04')
    fd.write(struct.pack('I',m.shape[0])) # rows
    fd.write('\04')
    fd.write(struct.pack('I',m.shape[1])) # cols
    # Data,
    m.tofile(fd, sep="") # binary
  finally:
    if fd is not file_or_fd : fd.close()



#################################################
# Confusion Network bins,
# Typically composed of tuples (words/phones/states, posteriors),
# (uses Posterior datatype from Kaldi)
#

def read_post_ark(file_or_fd):
  """ genrator(key,vec<vec<int,float>>) = read_post_ark(file)
   Returns generator of (key,posterior) tuples, which reads from ark file.
   file : filename or opened ark-file descriptor

   Hint, read scp to hash:
   d = dict((u,d) for u,d in pytel.kaldi_io.read_post_ark(file))
  """
  fd = open_or_fd(file_or_fd)
  try:
    key = read_key(fd)
    while key:
      post = read_post(fd)
      yield key, post
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_post(file_or_fd):
  """ [post] = read_post(file_or_fd)
   Reads kaldi Posterior in binary format. 
   
   Posterior is vec<vec<int,float>>, where outer-vector is over bins/frames, 
   inner vector is over words/phones/states, and inner-most tuple is composed 
   of an ID (integer) and POSTERIOR (float-value).
  """
  fd = open_or_fd(file_or_fd)
  ans=[]
  binary = fd.read(2); assert(binary == '\0B'); # binary flag
  assert(fd.read(1) == '\4'); # int-size
  outer_vec_size = struct.unpack('<i', fd.read(4))[0] # number of frames (or bins)
  for i in range(outer_vec_size):
    assert(fd.read(1) == '\4'); # int-size
    inner_vec_size = struct.unpack('<i', fd.read(4))[0] # number of records for frame (or bin)
    id = np.zeros(inner_vec_size, dtype=int) # buffer for integer id's
    post = np.zeros(inner_vec_size, dtype=float) # buffer for posteriors
    for j in range(inner_vec_size):
      assert(fd.read(1) == '\4'); # int-size
      id[j] = struct.unpack('<i', fd.read(4))[0] # id 
      assert(fd.read(1) == '\4'); # float-size
      post[j] = struct.unpack('<f', fd.read(4))[0] # post
    ans.append(zip(id,post))
  if fd is not file_or_fd: fd.close()
  return ans



#################################################
# Confusion Network begin/end times for the bins 
# (kaldi stores them separately), 
#

def read_cntime_ark(file_or_fd):
  """ genrator(key,vec<float,float>) = read_cntime_ark(file)
   Returns generator of (key,cntime) tuples, which are read from ark file.
   file_or_fd : filename or opened file-descriptor

   Hint, read scp to hash:
   d = dict((u,d) for u,d in pytel.kaldi_io.read_cntime_ark(file))
  """
  fd = open_or_fd(file_or_fd)
  try:
    key = read_key(fd)
    while key:
      cntime = read_cntime(fd)
      yield key, cntime
      key = read_key(fd)
  finally:
    if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd):
  """ [cntime] = read_cntime(file_or_fd)
   Reads structure representing begin/end times of bins in confusion network.
   Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'
   file_or_fd : filename or opened file-descriptor
  """
  fd = open_or_fd(file_or_fd)
  binary = fd.read(2); assert(binary == '\0B'); # assuming it's binary
  assert(fd.read(1) == '\4'); # int-size
  vec_size = struct.unpack('<i', fd.read(4))[0] # number of frames (or bins)
  t_beg = np.zeros(vec_size, dtype=float)
  t_end = np.zeros(vec_size, dtype=float)
  for i in range(vec_size):
    assert(fd.read(1) == '\4'); # float-size
    t_beg[i] = struct.unpack('<f', fd.read(4))[0] # begin-time of bin
    assert(fd.read(1) == '\4'); # float-size
    t_end[i] = struct.unpack('<f', fd.read(4))[0] # end-time of bin
  ans = zip(t_beg,t_end)
  if fd is not file_or_fd : fd.close()
  return ans

#################################################
# 'segments' file (used to select speech frames for Olda),
#

def read_segments_as_bool_vec(segments_file):
  """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
   using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
   - t-beg, t-end is in seconds, 
   - assumed 100 frames/second,
  """
  segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
  # Sanity checks,
  assert(len(segs) > 0) # empty segmentation is an error,
  assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
  # Convert time to frame-indexes,
  start = np.rint([100 * rec[2] for rec in segs]).astype(int)
  end = np.rint([100 * rec[3] for rec in segs]).astype(int)
  # Taken from 'read_lab_to_bool_vec', htk.py, 
  frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                   np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
  assert np.sum(end-start) == np.sum(frms)
  return frms


