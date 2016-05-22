#!/usr/bin/env python

import numpy as np
import struct, re
import scipy.io.wavfile as spiowav
import gzip
#import pytel.kaldi_io
import sys

# Reading,
def open_or_fd(file, mode='rb'):
  """ fd = open_or_fd(file)
   Open file (or gzipped file), or forward the file-descriptor.
  """
  try:
    if file.split('.')[-1] == 'gz':
      fd = gzip.open(file, mode)
    else:
      fd = open(file, mode)
  except AttributeError:
    fd = file
  return fd

def read_scp_line(line, return_key=False):
  """ generator(key,mat) = read_scp_line(line)
   Returns matrix in line of kaldi_scp file
   Returns generator of (key,matrix) tuples, which are read from kaldi scp file.
   file : filename or opened scp-file descriptor

   Hint, read scp to hash:
   d = dict((u,d) for u,d in pytel.kaldi_io.read_mat_scp(file))
  """
  line = line.rstrip()
  (key, aux) = line.split(' ')
  (ark, offset) = aux.split(':')
  with open(ark, 'rb') as f:
    f.seek(int(offset))
    #mat = pytel.kaldi_io.read_mat(f)
    mat = read_mat(f)
  #yield mat, key
  if return_key:
    return mat, key
  else:
    return mat


def read_mat(file_or_fd):
  """ [mat] = read_mat(file_or_fd)
   Reads kaldi matrix from file or file-descriptor, can be ascii or binary.
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



def write_stdout_ascii(mat, utt):
  # write to stdout in ascii kaldi_mat format
  (num_frms, fea_dim) = mat.shape
  # first line utt followed by [ 
  sys.stdout.write("%s  [ \n" %utt)
  for i in xrange(num_frms):
    sys.stdout.write("  ")
    for j in xrange(fea_dim):
      sys.stdout.write("%f " %mat[i,j])
    if i < num_frms-1:
      sys.stdout.write(" \n")        
    else:
      sys.stdout.write("]")
  sys.stdout.flush()


import Queue
from threading import Thread

class KaldiScpReader(object):
  @staticmethod
  def read_scp_reader_worker(scp_reader):
    for i, scp_line in enumerate(scp_reader._scp):
      scp_reader._queue.put((scp_reader._reader(read_scp_line(scp_line), scp_line.split()[0]), scp_line.split()[0]))
      if scp_reader._closed or scp_reader._max_records_to_read == i+1:
        break
    scp_reader._queue.put(None)
    #tar_reader._tar.close()

  def __init__(self, scp_file_name, reader=lambda xx, r: xx, max_records_to_read=None, queue_size=1000, reader_args=()):
    self._scp = open(scp_file_name, 'r')
    self._queue = Queue.Queue(maxsize=queue_size)
    self._closed = False
    self._reader = lambda xx, r: reader(xx, r, *reader_args) #xx is matrix, r is key
    self._max_records_to_read = max_records_to_read
    Thread(target=self.read_scp_reader_worker, args=(self,)).start()

  def __iter__(self):
    return self

  def __enter__(self):
    return self

  def next(self):
    i = self._queue.get()
    if i is None or self._closed:
      raise StopIteration
    return i

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    if not self._closed:
      self._closed = True
      self._scp.close()
      try:
        self._queue.get_nowait()
        self._queue.get_nowait() # to make sure that worker puts final None to queue and terminates
      except Queue.Empty as e:
        pass

# with mypytel.utils.KaldiScpReader('/mnt/matylda3/qmallidi/JHU2015/Kaldi/kaldi/egs/ami/s5_theano/data-fbank/ihm/train/feats.scp') as data_it:
#   for jj, (X, key) in enumerate(data_it):
#     print key, X.shape


