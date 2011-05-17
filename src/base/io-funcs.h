// base/io-funcs.h
// Copyright 2009-2011 Microsoft Corporation  Arnab Ghoshal  Jan Silovsky   Yanmin Qian

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_BASE_IO_FUNCS_H_
#define KALDI_BASE_IO_FUNCS_H_

// This header only contains some relatively low-level I/O functions.
// The full Kaldi I/O declarations are in ../util/kaldi-io.h
// and ../util/kaldi-table.h
// They were put in util/ in order to avoid making the Matrix library
// dependent on them.

#include <cctype>
#include <limits>
#include <vector>
#include <string>
#include "base/kaldi-common.h"

namespace kaldi {



/*
  This comment describes the Kaldi approach to I/O.  All objects can be written
  and read in two modes: binary and text.  In addition we want to make the I/O
  work if we redefine the typedef "BaseFloat" between floats and doubles.
  We also want to have control over whitespace in text mode without affecting
  the meaning of the file, for pretty-printing purposes.

  Errors are handled by throwing an exception (std::runtime_error).

  For integer and floating-point types (and boolean values):

   WriteBasicType(std::ostream &, bool binary, const T&);
   ReadBasicType(std::istream &, bool binary, T*);

  and we expect these functions to be defined in such a way that they work when
  the type T changes between float and double, so you can read float into double
  and vice versa].  Note that for efficiency and space-saving reasons, the Vector
  and Matrix classes do not use these functions [but they preserve the type
  interchangeability in their own way]

  For a class (or struct) C:
  class C {
  ..
    Write(std::ostream &, bool binary, [possibly extra optional args for specific classes]) const;
    Read(std::istream &, bool binary, [possibly extra optional args for specific classes]);
    // The only actual optional args we used are the "add" arguments in Vector/Matrix classes,
    // which specify whether we should sum the data already in the class with the data being
    // read.
  ..
  }

  For types which are typedef's involving stl classes, I/O is as follows:
  typedef std::vector<std::pair<A, B> > MyTypedefName;

  The user should define something like:

   WriteMyTypedefName(std::ostream &, bool binary, const MyTypedefName &t);
   ReadMyTypedefName(std::ostream &, bool binary, MyTypedefName *t);

  The user would have to write these functions.

  For a type std::vector<T>:

   void WriteIntegerVector(std::ostream &os, bool binary, const std::vector<T> &v);
   void ReadIntegerVector(std::istream &is, bool binary, std::vector<T> *v);

  For other types, e.g. vectors of pairs, the user should create a routine of the
  type WriteMyTypedefName.  This is to avoid introducing confusing templated functions;
  we could easily create templated functions to handle most of these cases but they
  would have to share the same name.

  It also often happens that the user needs to write/read special markers as part
  of a file.  These might be class headers, or separators/identifiers in the class.
  We provide special functions for manipulating these.  These special markers must
  be nonempty and must not contain any whitespace.

    void WriteMarker(std::ostream &os, bool binary, const char*);
    void WriteMarker(std::ostream &os, bool binary, const std::string & marker);
    int PeekMarker(std::istream &is, bool binary);
    void ReadMarker(std::istream &is, bool binary, std::string *str);
    void ReadMarker(std::istream &is, bool binary, char *buf, size_t bufsz)

  WriteMarker writes the marker and one space (whether in binary or text mode).

  PeekMarker returns the first character of the next marker, by consuming whitespace (in
  text mode) and then returning the peek() character.  It returns -1 at EOF; it doesn't throw.
  It's useful if a class can have various forms based on typedefs and virtual classes, and
  wants to know which version to read.  The two forms of ReadMarker allow the caller to
  obtain the next marker.

  There is currently no special functionality for writing/reading strings (where the strings
  contain data rather than "special markers" that are whitespace-free and nonempty).  This is
  because Kaldi is structured in such a way that strings don't appear, except as OpenFst symbol
  table entries (and these have their own format).


  NOTE: you should not call ReadIntegerType and WriteIntegerType with types,
  such as int and size_t, that are machine-independent -- at least not
  if you want your file formats to port between machines.  Use int32 and
  int64 where necessary.  There is no way to detect this using compile-time
  assertions because C++ only keeps track of the internal representation of
  the type.
*/

/// \addtogroup io_funcs_basic
/// @{


// WriteBasicType is the name of the write function
// for bool, integer types, and floating-point types.
// They all throw on error.
template<class T> void WriteBasicType(std::ostream &os, bool binary, T t);

template<class T> void ReadBasicType(std::istream &is, bool binary, T *t);


// Declare specialization for bool.
template<>
void WriteBasicType<bool>(std::ostream &os, bool binary, bool b);

template <>
void ReadBasicType<bool>(std::istream &is, bool binary, bool *b);


// Declare specializations for float and double.
template<>
void WriteBasicType<float>(std::ostream &os, bool binary, float f);

template<>
void WriteBasicType<double>(std::ostream &os, bool binary, double f);

template<>
void ReadBasicType<float>(std::istream &is, bool binary, float *f);

template<>
void ReadBasicType<double>(std::istream &is, bool binary, double *f);



// Template that covers integers.
template<class T>  void WriteBasicType(std::ostream &os,
                                       bool binary, T t) {
  // Compile time assertion that this is not called with a wrong type.
  KALDI_ASSERT_IS_INTEGER_TYPE(T);
  if (binary) {
    char len_c = (std::numeric_limits<T>::is_signed ? 1 :  -1)
        * static_cast<char>(sizeof(t));
    os.put(len_c);
    os.write(reinterpret_cast<const char *>(&t), sizeof(t));
  } else {
    if (sizeof(t) == 1)
      os << static_cast<int16>(t) << " ";
    else
      os << t << " ";
  }
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteBasicType.");
  }
}

// Template that covers integers.
template<class T> inline void ReadBasicType(std::istream &is,
                                            bool binary, T *t) {
#ifdef KALDI_PARANOID
  assert(t != NULL);
#endif
  // Compile time assertion that this is not called with a wrong type.
  KALDI_ASSERT_IS_INTEGER_TYPE(T);
  if (binary) {
    char len_c = is.get(), len_c_expected
        = (std::numeric_limits<T>::is_signed ? 1 :  -1)
        * static_cast<char>(sizeof(*t));

    if (len_c !=  len_c_expected) {
      KALDI_ERR << "ReadBasicType: did not get expected integer type, "
                << static_cast<int>(len_c)
                << " vs. " << static_cast<int>(len_c_expected)
                << ".  You can change this code to successfully"
                << " read it later, if needed.";
      // insert code here to read "wrong" type.  Might have a switch statement.
    }
    is.read(reinterpret_cast<char *>(t), sizeof(*t));
  } else {
    if (sizeof(*t) == 1) {
      int16 i;
      is >> i;
      *t = i;
    } else {
      is >> *t;
    }
  }
  if (is.fail()) {
    KALDI_ERR << "Read failure in ReadBasicType, file position is "
              << is.tellg() << ", next char is " << is.peek();
  }
}


template<class T> inline void WriteIntegerVector(std::ostream &os, bool binary,
                                                 const std::vector<T> &v) {
  // Compile time assertion that this is not called with a wrong type.
  KALDI_ASSERT_IS_INTEGER_TYPE(T);
  if (binary) {
    char sz = sizeof(T);  // this is currently just a check.
    os.write(&sz, 1);
    int32 vecsz = static_cast<int32>(v.size());
    assert((size_t)vecsz == v.size());
    os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
    if (vecsz != 0) {
      os.write(reinterpret_cast<const char *>(&(v[0])), sizeof(T)*vecsz);
    }
  } else {
    // focus here is on prettiness of text form rather than
    // efficiency of reading-in.
    // reading-in is dominated by low-level operations anyway:
    // for efficiency use binary.
    os << "[ ";
    typename std::vector<T>::const_iterator iter = v.begin(), end = v.end();
    for (; iter != end; ++iter) {
      if (sizeof(T) == 1)
        os << static_cast<int16>(*iter) << " ";
      else
        os << *iter << " ";
    }
    os << "]\n";
  }
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteIntegerType.");
  }
}


template<class T> inline void ReadIntegerVector(std::istream &is,
                                                bool binary,
                                                std::vector<T> *v) {
  KALDI_ASSERT_IS_INTEGER_TYPE(T);
  assert(v != NULL);
  if (binary) {
    int sz = is.peek();
    if (sz == sizeof(T)) {
      is.get();
    } else {  // this is currently just a check.
      KALDI_ERR << "ReadIntegerVector: expected to see type of size "
                << sizeof(T) << ", saw instead " << sz << ", at file position "
                << is.tellg();
    }
    int32 vecsz;
    is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
    if (is.fail() || vecsz < 0) goto bad;
    v->resize(vecsz);
    if (vecsz > 0) {
      is.read(reinterpret_cast<char *>(&((*v)[0])), sizeof(T)*vecsz);
    }
  } else {
    std::vector<T> tmp_v;  // use temporary so v doesn't use extra memory
                           // due to resizing.
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      KALDI_ERR << "ReadIntegerVector: expected to see [, saw "
                << is.peek() << ", at file position " << is.tellg();
    }
    is.get();  // consume the '['.
    is >> std::ws;  // consume whitespace.
    while (is.peek() != static_cast<int>(']')) {
      if (sizeof(T) == 1) {  // read/write chars as numbers.
        int16 next_t;
        is >> next_t >> std::ws;
        if (is.fail()) goto bad;
        else
            tmp_v.push_back((T)next_t);
      } else {
        T next_t;
        is >> next_t >> std::ws;
        if (is.fail()) goto bad;
        else
            tmp_v.push_back(next_t);
      }
    }
    is.get();  // get the final ']'.
    *v = tmp_v;  // could use std::swap to use less temporary memory, but this
    // uses less permanent memory.
  }
  if (!is.fail()) return;
 bad:
  KALDI_ERR << "ReadIntegerVector: read failure at file position "
            << is.tellg();
}

/// The WriteMarker functions are for writing nonempty
/// sequences of non-space characters.
/// They are not for general strings.
void WriteMarker(std::ostream &os, bool binary, const char *marker);
void WriteMarker(std::ostream &os, bool binary, const std::string & marker);

/// PeekMarker consumes whitespace (if binary == false) and then returns
/// the peek() value of the stream.
int PeekMarker(std::istream &is, bool binary);

/// ReadMarker gets the next marker and puts it in str (exception on failure).
void ReadMarker(std::istream &is, bool binary, std::string *marker);

/// ExpectMarker tries to read in the given marker, and throws an exception
/// on failure.
void ExpectMarker(std::istream &is, bool binary, const char *marker);
void ExpectMarker(std::istream &is, bool binary, const std::string & marker);

/// ExpectPretty attempts to read the text in "marker", but only in non-binary
/// mode.  Throws exception on failure.  It expects an exact match except that
/// arbitrary whitespace matches arbitrary whitespace.
void ExpectPretty(std::istream &is, bool binary, const char *marker);
void ExpectPretty(std::istream &is, bool binary, const std::string & marker);

/// @} end "addtogroup io_funcs_basic"


/// InitKaldiOutputStream initializes an opened stream for writing by writing an
/// optional binary header and modifying the floating-point precision; it will
/// typically not be called by users directly.
inline void InitKaldiOutputStream(std::ostream &os, bool binary) {
  // This does not throw exceptions (does not check for errors).
  if (binary) {
    os.put('\0');
    os.put('B');
  }
  // Note, in non-binary mode we may at some point want to mess with
  // the precision a bit.
  // 7 is a bit more than the precision of float..
  if (os.precision() < 7)
    os.precision(7);
}

/// InitKaldiInputStream initializes an opened stream for reading by detecting
/// the binary header and setting the "binary" value appropriately;
/// It will typically not be called by users directly.
inline bool InitKaldiInputStream(std::istream &is, bool *binary) {
  // Sets the 'binary' variable.
  // Throws exception in the very unusual situation that stream
  // starts with '\0' but not then 'B'.

  if (is.peek() == '\0') {  // seems to be binary
    is.get();
    if (is.peek() != 'B') {
      return false;
    }
    is.get();
    *binary = true;
    return true;
  } else {
    *binary = false;
    return true;
  }
}



}  // end namespace kaldi.

#endif  // KALDI_BASE_IO_FUNCS_H_
