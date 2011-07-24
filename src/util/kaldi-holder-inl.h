// util/kaldi-holder-inl.h

// Copyright 2009-2011     Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_UTIL_KALDI_HOLDER_INL_H_
#define KALDI_UTIL_KALDI_HOLDER_INL_H_

#include <algorithm>
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

/// \addtogroup holders
/// @{


// KaldiObjectHolder is valid only for Kaldi objects with
// copy constructors, default constructors, and "normal"
// Kaldi Write and Read functions.  E.g. it works for
// Matrix and Vector.
template<class KaldiType> class KaldiObjectHolder {
 public:
  typedef KaldiType T;

  KaldiObjectHolder(): t_(NULL) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      t.Write(os, binary);
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object: " << e.what();
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() {
    if (t_) {
      delete t_;
      t_ = NULL;
    }
  }

  // Reads into the holder.
  bool Read(std::istream &is) {
    if (t_) delete t_;
    t_ = new T;
    // Don't want any existing state to complicate the read functioN: get new object.
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object, failed reading binary header\n";
      return false;
    }
    try {
      t_->Read(is, is_binary);
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object ";
	  if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      delete t_;
      t_ = NULL;
      return false;
    }
  }

  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {
    // code error if !t_.
    if (!t_) KALDI_ERR << "KaldiObjectHolder::Value() called wrongly.";
    return *t_;
  }

  ~KaldiObjectHolder() { if (t_) delete t_; }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiObjectHolder);
  T *t_;
};


// BasicHolder is valid for float, double, bool, and integer
// types.  There will be a compile time error otherwise, because
// we make sure that the {Write, Read}BasicType functions do not
// get instantiated for other types.

template<class BasicType> class BasicHolder {
 public:
  typedef BasicType T;

  BasicHolder(): t_(static_cast<T>(-1)) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      WriteBasicType(os, binary, t);
      if (!binary) os << '\n';  // Makes output format more readable and
      // easier to manipulate.
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object: " << e.what();
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() { }

  // Reads into the holder.
  bool Read(std::istream &is) {
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    try {
      int c;
      if (!is_binary) {  // This is to catch errors, the class would work without it..
        // Eat up any whitespace and make sure it's not newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) is.get();
        if (is.peek() == '\n') {
          KALDI_WARN << "Found newline but expected basic type.";
          return false;  // This is just to catch a more-
          // likely-than average type of error (empty line before the token), since
          // ReadBasicType will eat it up.
        }
      }

      ReadBasicType(is, is_binary, &t_);

      if (!is_binary) {  // This is to catch errors, the class would work without it..
        // make sure there is a newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) is.get();
        if (is.peek() != '\n') {
          KALDI_WARN << "BasicHolder::Read, expected newline, got "
                     << CharToString(is.peek()) << ", position " << is.tellg();
          return false;
        }
        is.get();  // Consume the newline.
      }
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {
    return t_;
  }

  ~BasicHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicHolder);

  T t_;
};


/// A Holder for a vector of basic types, e.g.
/// std::vector<int32>, std::vector<float>, and so on.
/// Note: a basic type is defined as a type for which ReadBasicType
/// and WriteBasicType are implemented, i.e. integer and floating
/// types, and bool.
template<class BasicType> class BasicVectorHolder {
 public:
  typedef std::vector<BasicType> T;

  BasicVectorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  // need to write the size, in binary mode.
        KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(t.size())) == t.size());
        // Or this Write routine cannot handle such a large vector.
        // use int32 because it's fixed size regardless of compilation.
        // change to int64 (plus in Read function) if this becomes a problem.
        WriteBasicType(os, binary, static_cast<int32>(t.size()));
        for (typename std::vector<BasicType>::const_iterator iter = t.begin();
            iter != t.end(); ++iter)
          WriteBasicType(os, binary, *iter);

      } else {
        for (typename std::vector<BasicType>::const_iterator iter = t.begin();
            iter != t.end(); ++iter)
          WriteBasicType(os, binary, *iter);
        os << '\n';  // Makes output format more readable and
        // easier to manipulate.  In text mode, this function writes something like
        // "1 2 3\n".
      }
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object (BasicVector). ";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    if (!is_binary) {
      // In text mode, we terminate with newline.
      std::string line;
      getline(is, line);  // this will discard the \n, if present.
      if (is.fail()) {
        KALDI_WARN << "BasicVectorHolder::Read, error reading line " << (is.eof() ? "[eof]" : "");
        return false;  // probably eof.  fail in any case.
      }
      std::istringstream line_is(line);
      try {
        while (1) {
          line_is >> std::ws;  // eat up whitespace.
          if (line_is.eof()) break;
          BasicType bt;
          ReadBasicType(line_is, false, &bt);
          t_.push_back(bt);
        }
        return true;
      } catch(std::exception &e) {
        KALDI_WARN << "BasicVectorHolder::Read, could not interpret line: " << line;
        if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
        return false;
      }
    } else {  // binary mode.
      size_t filepos = is.tellg();
      try {
        int32 size;
        ReadBasicType(is, true, &size);
        t_.resize(size);
        for (typename std::vector<BasicType>::iterator iter = t_.begin();
            iter != t_.end();
            ++iter) {
          ReadBasicType(is, true, &(*iter));
        }
        return true;
      } catch (...) {
        KALDI_WARN << "BasicVectorHolder::Read, read error or unexpected data at archive entry beginning at file position " << filepos;
        return false;
      }
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {  return t_; }

  ~BasicVectorHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicVectorHolder);
  T t_;
};


/// BasicVectorVectorHolder is a Holder for a vector of vector of
/// a basic type, e.g. std::vector<int32>, std::vector<float>, and so on.
/// Note: a basic type is defined as a type for which ReadBasicType
/// and WriteBasicType are implemented, i.e. integer and floating
/// types, and bool.
template<class BasicType> class BasicVectorVectorHolder {
 public:
  typedef std::vector<std::vector<BasicType> > T;

  BasicVectorVectorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  // need to write the size, in binary mode.
        KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(t.size())) == t.size());
        // Or this Write routine cannot handle such a large vector.
        // use int32 because it's fixed size regardless of compilation.
        // change to int64 (plus in Read function) if this becomes a problem.
        WriteBasicType(os, binary, static_cast<int32>(t.size()));
        for (typename std::vector<std::vector<BasicType> >::const_iterator iter = t.begin();
            iter != t.end(); ++iter) {
          KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(iter->size())) == iter->size());
          WriteBasicType(os, binary, static_cast<int32>(iter->size()));
          for (typename std::vector<BasicType>::const_iterator iter2=iter->begin();
              iter2 != iter->end(); ++iter2) {
            WriteBasicType(os, binary, *iter2);
          }
        }
      } else {  // text mode...
        // In text mode, we write out something like (for integers):
        // "1 2 3 ; 4 5 ; 6 ; ; 7 8 9 ;\n"
        // where the semicolon is a terminator (not a separator, which would lead
        // to difficulties representing an empty last element).
        for (typename std::vector<std::vector<BasicType> >::const_iterator iter = t.begin();
            iter != t.end();
            ++iter) {
          for (typename std::vector<BasicType>::const_iterator iter2=iter->begin();
              iter2 != iter->end(); ++iter2)
            WriteBasicType(os, binary, *iter2);
          os << "; ";
        }
        os << '\n';
      }
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object (BasicVector). ";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary header\n";
      return false;
    }
    if (!is_binary) {
      // In text mode, we terminate with newline.
      try {  // catching errors from ReadBasicType..
        std::vector<BasicType> v;  // temporary vector
        while (1) {
          int i = is.peek();
          if (i == -1) {
            KALDI_WARN << "BasicVectorHolder::Read, unexpected EOF";
            return false;
          } else if (static_cast<char>(i) == '\n') {
            if (!v.empty()) {
              KALDI_WARN << "BasicVectorHolder::Read, no semicolon before newline (wrong format)";
              return false;
            } else { is.get(); return true; }
          } else if (std::isspace(i)) {
            is.get();
          } else if (static_cast<char>(i) == ';') {
            t_.push_back(v);
            v.clear();
            is.get();
          } else {  // some object we want to read...
            BasicType b;
            ReadBasicType(is, false, &b);  // throws on error.
            v.push_back(b);
          }
        }
      } catch(std::exception &e) {
        KALDI_WARN << "BasicVectorVectorHolder::Read, read error";
        if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
        return false;
      }
    } else {  // binary mode.
      size_t filepos = is.tellg();
      try {
        int32 size;
        ReadBasicType(is, true, &size);
        t_.resize(size);
        for (typename std::vector<std::vector<BasicType> >::iterator iter = t_.begin();
            iter != t_.end();
            ++iter) {
          int32 size2;
          ReadBasicType(is, true, &size2);
          iter->resize(size2);
          for (typename std::vector<BasicType>::iterator iter2 = iter->begin();
              iter2 != iter->end();
              ++iter2)
            ReadBasicType(is, true, &(*iter2));
        }
        return true;
      } catch (...) {
        KALDI_WARN << "BasicVectorHolder::Read, read error or unexpected data at archive entry beginning at file position " << filepos;
        return false;
      }
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {  return t_; }

  ~BasicVectorVectorHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicVectorVectorHolder);
  T t_;
};



// We define a Token as a nonempty, printable, whitespace-free std::string.
// The binary and text formats here are the same (newline-terminated)
// and as such we don't bother with the binary-mode headers.
class TokenHolder {
 public:
  typedef std::string T;

  TokenHolder() {}

  static bool Write(std::ostream &os, bool, const T &t) {  // ignore binary-mode.
    KALDI_ASSERT(IsToken(t));
    os << t << '\n';
    return os.good();
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    is >> t_;
    if (is.fail()) return false;
    char c;
    while (isspace(c = is.peek()) && c!= '\n') is.get();
    if (is.peek() != '\n') {
      KALDI_ERR << "TokenHolder::Read, expected newline, got char " << CharToString(is.peek())
                << ", at stream pos " << is.tellg();
      return false;
    }
    is.get();  // get '\n'
    return true;
  }


  // Since this is fundamentally a text format, read in text mode (would work
  // fine either way, but doing it this way will exercise more of the code).
  static bool IsReadInBinary() { return false; }

  const T &Value() const { return t_; }

  ~TokenHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TokenHolder);
  T t_;
};

// A Token is a nonempty, whitespace-free std::string.
// Class TokenVectorHolder is a Holder class for vectors of these.
class TokenVectorHolder {
 public:
  typedef std::vector<std::string> T;

  TokenVectorHolder() { }

  static bool Write(std::ostream &os, bool, const T &t) {  // ignore binary-mode.
    for (std::vector<std::string>::const_iterator iter = t.begin();
        iter != t.end();
        ++iter) {
      KALDI_ASSERT(IsToken(*iter));  // make sure it's whitespace-free, printable and nonempty.
      os << *iter << ' ';
    }
    os << '\n';
    return os.good();
  }

  void Clear() { t_.clear(); }


  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();

    // there is no binary/non-binary mode.

    std::string line;
    getline(is, line);  // this will discard the \n, if present.
    if (is.fail()) {
      KALDI_WARN << "BasicVectorHolder::Read, error reading line " << (is.eof() ? "[eof]" : "");
      return false;  // probably eof.  fail in any case.
    }
    const char* white_chars = " \t\n\r\f\v";
    SplitStringToVector(line, white_chars, &t_, true);  // true== omit empty strings e.g.
    // between spaces.
    return true;
  }

  // Read in text format since it's basically a text-mode thing.. doesn't really matter,
  // it would work either way since we ignore the extra '\r'.
  static bool IsReadInBinary() { return false; }

  const T &Value() const { return t_; }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TokenVectorHolder);
  T t_;
};


class HtkMatrixHolder {
 public:
  typedef std::pair<Matrix<BaseFloat>, HtkHeader> T;

  HtkMatrixHolder() {}

  static bool Write(std::ostream &os, bool binary, const T &t) {
    if (!binary)
      KALDI_ERR << "Non-binary HTK-format write not supported.";
    bool ans = WriteHtk(os, t.first, t.second);
    if (!ans)
      KALDI_WARN << "Error detected writing HTK-format matrix.";
    return ans;
  }

  void Clear() { t_.first.Resize(0, 0); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    bool ans = ReadHtk(is, &t_.first, &t_.second);
    if (!ans) {
      KALDI_WARN << "Error detected reading HTK-format matrix.";
      return false;
    }
    return ans;
  }

  // HTK-format matrices only read in binary.
  static bool IsReadInBinary() { return true; }

  const T &Value() const { return t_; }


  // No destructor.
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(HtkMatrixHolder);
  T t_;
};


// PosteriorHolder is a holder for Posterior, which is
// std::vector<std::vector<std::pair<int32, BaseFloat> > >
// This is used for storing posteriors of transition id's for an
// utterance.
class PosteriorHolder {
 public:
  typedef Posterior T;

  PosteriorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {
        int32 sz = t.size();
        WriteBasicType(os, binary, sz);
        for (Posterior::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
          int32 sz2 = iter->size();
          WriteBasicType(os, binary, sz2);
          for (std::vector<std::pair<int32, BaseFloat> >::const_iterator iter2=iter->begin();
              iter2 != iter->end();
              iter2++) {
            WriteBasicType(os, binary, iter2->first);
            WriteBasicType(os, binary, iter2->second);
          }
        }
      } else {  // In text-mode, choose a human-friendly, script-friendly format.
        // format is [ 1235 0.6 12 0.4 ] [ 34 1.0 ] ...
        // We could have used the same code as in the binary case above,
        // but this would have resulted in less readable output.
        for (Posterior::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
          os << "[ ";
          for (std::vector<std::pair<int32, BaseFloat> >::const_iterator iter2=iter->begin();
              iter2 != iter->end();
              iter2++) {
            os << iter2->first << ' ' << iter2->second << ' ';
          }
          os << "] ";
        }
        os << '\n';  // newline terminate the record.
      }
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing table of posteriors";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() {  // free up memory and clear t_.
    Posterior tmp;
    std::swap(tmp, t_);
  }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();

    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object, failed reading binary header\n";
      return false;
    }
    try {
      if (is_binary) {
        int32 sz;
        ReadBasicType(is, true, &sz);
        if (sz < 0)
          KALDI_ERR << "Reading posteriors: got negative size\n";
        t_.resize(sz);
        for (Posterior::iterator iter = t_.begin(); iter != t_.end(); ++iter) {
          int32 sz2;
          ReadBasicType(is, true, &sz2);
          if (sz2 < 0)
            KALDI_ERR << "Reading posteriors: got negative size\n";
          iter->resize(sz2);
          for (std::vector<std::pair<int32, BaseFloat> >::iterator iter2=iter->begin();
              iter2 != iter->end();
              iter2++) {
            ReadBasicType(is, true, &(iter2->first));
            ReadBasicType(is, true, &(iter2->second));
          }
        }
      } else {
        std::string line;
        getline(is, line);  // this will discard the \n, if present.
        if (is.fail()) {
          KALDI_WARN << "holder of Posterior: error reading line " << (is.eof() ? "[eof]" : "");
          return false;  // probably eof.  fail in any case.
        }
        std::istringstream line_is(line);
        while (1) {
          std::string str;
          line_is >> std::ws;  // eat up whitespace.
          if (line_is.eof()) break;
          line_is >> str;
          if (str != "[") KALDI_ERR << "Reading Posterior object: expecting [, got "
                                   << str << " (if this is an integer, possibly "
              "you gave alignments in place of posteriors?)";
          std::vector<std::pair<int32, BaseFloat> > this_vec;
          while (1) {
            line_is >> std::ws;
            if (line_is.peek() == ']') {
              line_is.get();
              break;
            }
            int32 i; BaseFloat p;
            line_is >> i >> p;
            if (line_is.fail())
              KALDI_ERR << "Error reading Posterior object (could not get data after \"[\");";
            this_vec.push_back(std::make_pair(i, p));
          }
          t_.push_back(this_vec);
        }
      }
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading table of posteriors";
	  if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      t_.clear();
      return false;
    }
  }

  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {
    return t_;
  }

  // No destructor.
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(PosteriorHolder);
  T t_;
};


// GauPostHolder is a holder for GauPost, which is
// std::vector<std::vector<std::pair<int32, Vector<BaseFloat> > > >
// This is used for storing posteriors of transition id's for an
// utterance.
class GauPostHolder {
 public:
  typedef GauPost T;

  GauPostHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      // We don't bother making this a one-line format.
      int32 sz = t.size();
      WriteBasicType(os, binary, sz);
      for (GauPost::const_iterator iter = t.begin(); iter != t.end(); ++iter) {
        int32 sz2 = iter->size();
        WriteBasicType(os, binary, sz2);
        for (std::vector<std::pair<int32, Vector<BaseFloat> > >::const_iterator iter2=iter->begin();
            iter2 != iter->end();
            iter2++) {
          WriteBasicType(os, binary, iter2->first);
          iter2->second.Write(os, binary);
        }
      }
      if(!binary) os << '\n';
      return os.good();
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught writing table of posteriors";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;  // Write failure.
    }
  }

  void Clear() {  // free up memory and clear t_.
    GauPost tmp;
    std::swap(tmp, t_);
  }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();

    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object, failed reading binary header\n";
      return false;
    }
    try {
      int32 sz;
      ReadBasicType(is, is_binary, &sz);
      if (sz < 0)
        KALDI_ERR << "Reading posteriors: got negative size\n";
      t_.resize(sz);
      for (GauPost::iterator iter = t_.begin(); iter != t_.end(); ++iter) {
        int32 sz2;
        ReadBasicType(is, is_binary, &sz2);
        if (sz2 < 0)
          KALDI_ERR << "Reading posteriors: got negative size\n";
        iter->resize(sz2);
        for (std::vector<std::pair<int32, Vector<BaseFloat> > >::iterator
                iter2=iter->begin();
            iter2 != iter->end();
            iter2++) {
          ReadBasicType(is, is_binary, &(iter2->first));
          iter2->second.Read(is, is_binary);
        }
      }
      return true;
    } catch (std::exception &e) {
      KALDI_WARN << "Exception caught reading table of posteriors";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      t_.clear();
      return false;
    }
  }

  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  const T &Value() const {
    return t_;
  }

  // No destructor.
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(GauPostHolder);
  T t_;
};

/// @} end "addtogroup holders"

} // end namespace kaldi



#endif
