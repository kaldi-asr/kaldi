// $Id: mfstream.h 383 2010-04-23 15:29:28Z nicolabertoldi $

/******************************************************************************
IrstLM: IRST Language Model Toolkit, compile LM
Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <streambuf>
#include <cstdio>

using namespace std;

#ifndef MF_STREAM_H
#define MF_STREAM_H

extern "C" {
  ssize_t write (int fd, const void* buf, size_t num);
  ssize_t read (int fd,  void* buf, size_t num);
  FILE *popen(const char *command, const char *type);
  int pclose(FILE *stream);
  int fseek( FILE *stream, long offset, int whence);
  long ftell( FILE *stream);
};


//! File description for I/O stream buffer
class fdbuf : public std::streambuf
{

protected:
  int fd;    // file descriptor

  // write one character
  virtual int_type overflow (int_type c) {
    char z = c;
    if (c != EOF) {
      if (write (fd, &z, 1) != 1) {
        return EOF;
      }
    }
    //cerr << "overflow: \n";
    //cerr << "pptr: " << (int) pptr() << "\n";
    return c;
  }

  // write multiple characters
  virtual
  std::streamsize xsputn (const char* s,
                          std::streamsize num) {
    return write(fd,s,num);

  }

  virtual streampos seekpos ( streampos /* unused parameter: sp */, ios_base::openmode /* unused parameter: which */= ios_base::in | ios_base::out ) {
    std::cerr << "mfstream::seekpos is not implemented" << std::endl;;

    return (streampos) 0;
  }

  //read one character
  virtual int_type underflow () {
    // is read position before end of buffer?
    if (gptr() < egptr()) {
      return traits_type::to_int_type(*gptr());
    }

    /* process size of putback area
     * - use number of characters read
     * - but at most four
     */
    int numPutback;
    numPutback = gptr() - eback();
    if (numPutback > 4) {
      numPutback = 4;
    }

    /* copy up to four characters previously read into
     * the putback buffer (area of first four characters)
     */
    std::memmove (buffer+(4-numPutback), gptr()-numPutback,
                  numPutback);

    // read new characters
    int num;
    num = read (fd, buffer+4, bufferSize-4);
    if (num <= 0) {
      // ERROR or EOF
      return EOF;
    }

    // reset buffer pointers
    setg (buffer+(4-numPutback),   // beginning of putback area
          buffer+4,                // read position
          buffer+4+num);           // end of buffer

    // return next character
    return traits_type::to_int_type(*gptr());
  }


  // read multiple characters
  virtual
  std::streamsize xsgetn (char* s,
                          std::streamsize num) {
    return read(fd,s,num);
  }

  static const int bufferSize = 10;    // size of the data buffer
  char buffer[bufferSize];             // data buffer

public:

  // constructor
  fdbuf (int _fd) : fd(_fd) {
    setg (buffer+4,     // beginning of putback area
          buffer+4,     // read position
          buffer+4);    // end position
  }

};



//! Extension of fstream to commands

class mfstream : public std::fstream
{

protected:
  fdbuf* buf;
  int _cmd;
  openmode _mode;
  FILE* _FILE;
	
  char _cmdname[500];

  int swapbytes(char *p, int sz, int n);

public:
	
  //! Creates and opens a file/command stream without a specified nmode
  mfstream () : std::fstream(), buf(NULL), _cmd(0), _FILE(NULL) {
    _cmdname[0]='\0';
  }
	
  //! Creates and opens a  file/command stream in a specified nmode
  mfstream (const char* name,openmode mode) : std::fstream() {
    _cmdname[0]='\0';
    _mode=mode;
    open(name,mode);
  }

  //! Closes and destroys a file/command stream
  ~mfstream() {
    if (_cmd<2) close();
  }

  //! Opens an existing mfstream
  void open(const char *name,openmode mode);

  //! Closes an existing mfstream
  void close();

  //! Write function for machine-independent byte order
  mfstream& writex(void *p, int sz,int n=1);

  //! Read function for machine-independent byte order
  mfstream& readx(void *p, int sz,int n=1);

  //! Write function at a given stream position for machine-independent byte order
  mfstream& iwritex(streampos loc,void *ptr,int size,int n=1);

  //! Tells current position within a file
  streampos tellp();

  //! Seeks a position within a file
  mfstream& seekp(streampos loc);

  //! Reopens an input stream
  mfstream& reopen();
};

class inputfilestream : public std::istream
{
protected:
	std::streambuf *m_streambuf;
	bool _good;
public:
	
	inputfilestream(const std::string &filePath);
	~inputfilestream();
	inline bool good() { return _good; }
	void close();
};

#endif
