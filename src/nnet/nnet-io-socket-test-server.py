#!/bin/env python

import SocketServer
import io


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


def read_vec_flt(file_or_fd):
  """ [flt-vec] = read_vec_flt(file_or_fd)
   Read float vector from file or file-descriptor, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd)
  binary = fd.read(2)
  if binary == '\0B': # binary flag
    assert(fd.read(1) == '\4'); # int-size
    vec_size = struct.unpack('<i', fd.read(4))[0] # vector dim
    ans = np.zeros(vec_size, dtype=float)
    for i in range(vec_size):
      assert(fd.read(1) == '\4'); # float-size
      ans[i] = struct.unpack('<f', fd.read(4))[0] #data
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





class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        msg_size = struct.unpack('<q', self.request.recv(8))[0]
        remain_size = msg_size`
        
        

        self.data =  io.BytesIO(self.request.recv(8)).read(8)
        ans[i] = struct.unpack('<f', fd.read(4))[0] #data
        
        print "{} wrote:".format(self.client_address[0])
        print self.data
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())

if __name__ == "__main__":
    HOST, PORT = "localhost", 12345

    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
