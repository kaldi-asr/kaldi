// $Id: gzfilebuf.h 236 2009-02-03 13:25:19Z nicolabertoldi $

#ifndef _GZFILEBUF_H_
#define _GZFILEBUF_H_

#include <cstdio>
#include <streambuf>
#include <cstring>
#include <zlib.h>
#include <iostream>

class gzfilebuf : public std::streambuf
{
public:
  gzfilebuf(const char *filename) {
    _gzf = gzopen(filename, "rb");
    setg (_buff+sizeof(int),     // beginning of putback area
          _buff+sizeof(int),     // read position
          _buff+sizeof(int));    // end position
  }
  ~gzfilebuf() {
    gzclose(_gzf);
  }
protected:
  virtual int_type overflow (int_type /* unused parameter: c */) {
    std::cerr << "gzfilebuf::overflow is not implemented" << std::endl;;
    throw;
  }

  // write multiple characters
  virtual std::streamsize xsputn (const char* /* unused parameter: s */, std::streamsize /* unused parameter: num */) {
    std::cerr << "gzfilebuf::xsputn is not implemented" << std::endl;;
    throw;
  }

  virtual std::streampos seekpos ( std::streampos /* unused parameter: sp */, std::ios_base::openmode /* unused parameter: which */= std::ios_base::in | std::ios_base::out ) {
    std::cerr << "gzfilebuf::seekpos is not implemented" << std::endl;;
    throw;
  }

  //read one character
  virtual int_type underflow () {
    // is read position before end of _buff?
    if (gptr() < egptr()) {
      return traits_type::to_int_type(*gptr());
    }

    /* process size of putback area
     * - use number of characters read
     * - but at most four
     */
    unsigned int numPutback = gptr() - eback();
    if (numPutback > sizeof(int)) {
      numPutback = sizeof(int);
    }

    /* copy up to four characters previously read into
     * the putback _buff (area of first four characters)
     */
    std::memmove (_buff+(sizeof(int)-numPutback), gptr()-numPutback,
                  numPutback);

    // read new characters
    int num = gzread(_gzf, _buff+sizeof(int), _buffsize-sizeof(int));
    if (num <= 0) {
      // ERROR or EOF
      return EOF;
    }

    // reset _buff pointers
    setg (_buff+(sizeof(int)-numPutback),   // beginning of putback area
          _buff+sizeof(int),                // read position
          _buff+sizeof(int)+num);           // end of buffer

    // return next character
    return traits_type::to_int_type(*gptr());
  }

  std::streamsize xsgetn (char* s,
                          std::streamsize num) {
    return gzread(_gzf,s,num);
  }

private:
  gzFile _gzf;
  static const unsigned int _buffsize = 1024;
  char _buff[_buffsize];
};

#endif
