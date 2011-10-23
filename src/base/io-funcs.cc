// base/io-funcs.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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

#include "base/io-funcs.h"
#include "base/kaldi-math.h"

namespace kaldi {

template<>
void WriteBasicType<bool>(std::ostream &os, bool binary, bool b) {
  os << (b ? "T":"F");
  if (!binary) os << " ";
  if (os.fail())
    KALDI_ERR << "Write failure in WriteBasicType<bool>";
}

template<>
void ReadBasicType<bool>(std::istream &is, bool binary, bool *b) {
  KALDI_PARANOID_ASSERT(b != NULL);
  if (!binary) is >> std::ws;  // eat up whitespace.
  char c = is.peek();
  if (c == 'T') {
      *b = true;
      is.get();
  } else if (c == 'F') {
      *b = false;
      is.get();
  } else {
    KALDI_ERR << "Read failure in ReadBasicType<bool>, file position is "
              << is.tellg() << ", next char is " << CharToString(c);
  }
}

template<>
void WriteBasicType<float>(std::ostream &os, bool binary, float f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char *>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

template<>
void WriteBasicType<double>(std::ostream &os, bool binary, double f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char *>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

template<>
void ReadBasicType<float>(std::istream &is, bool binary, float *f) {
  KALDI_PARANOID_ASSERT(f != NULL);
  if (binary) {
    double d;
    int c = is.peek();
    if (c == sizeof(*f)) {
      is.get();
      is.read(reinterpret_cast<char*>(f), sizeof(*f));
    } else if (c == sizeof(d)) {
      ReadBasicType(is, binary, &d);
      *f = d;
    } else {
      KALDI_ERR << "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *f;
  }
  if (is.fail()) {
    KALDI_ERR << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

template<>
void ReadBasicType<double>(std::istream &is, bool binary, double *d) {
  KALDI_PARANOID_ASSERT(d != NULL);
  if (binary) {
    float f;
    int c = is.peek();
    if (c == sizeof(*d)) {
      is.get();
      is.read(reinterpret_cast<char*>(d), sizeof(*d));
    } else if (c == sizeof(f)) {
      ReadBasicType(is, binary, &f);
      *d = f;
    } else {
      KALDI_ERR << "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *d;
  }
  if (is.fail()) {
    KALDI_ERR << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

void CheckMarker(const char *marker) {
  assert(*marker != '\0');  // check it's nonempty.
  while (*marker != '\0') {
    assert(!::isspace(*marker));
    marker++;
  }
}

void WriteMarker(std::ostream &os, bool binary, const char *marker) {
  // binary mode is ignored;
  // we use space as termination character in either case.
  assert(marker != NULL);
  CheckMarker(marker);  // make sure it's valid (can be read back)
  os << marker << " ";
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteMarker.");
  }
}

int Peek(std::istream &is, bool binary) {
  if (!binary) is >> std::ws;  // eat up whitespace.
  return is.peek();
}

void WriteMarker(std::ostream &os, bool binary, const std::string & marker) {
  WriteMarker(os, binary, marker.c_str());
}

void ReadMarker(std::istream &is, bool binary, std::string *str) {
  assert(str != NULL);
  if (!binary) is >> std::ws;  // consume whitespace.
  is >> *str;
  if (is.fail()) {
    KALDI_ERR << "ReadMarker, failed to read marker at file position "
              << is.tellg();
  }
  if (!isspace(is.peek())) {
    KALDI_ERR << "ReadMarker, expected space after marker, saw instead "
              << static_cast<char>(is.peek())
              << ", at file position " << is.tellg();
  }
  is.get();  // consume the space.
}


void PeekMarker(std::istream &is, bool binary, std::string *str) {
  assert(str != NULL);
  if (!binary) is >> std::ws;  // consume whitespace.
  std::streampos beg = is.tellg();
  is >> *str;
  if (is.fail()) {
    KALDI_ERR << "PeekMarker, failed to read marker at file position "
              << is.tellg();
  }
  is.seekg(beg);
}


void ExpectMarker(std::istream &is, bool binary, const char *marker) {
  int pos_at_start = is.tellg();
  assert(marker != NULL);
  CheckMarker(marker);  // make sure it's valid (can be read back)
  if (!binary) is >> std::ws;  // consume whitespace.
  std::string str;
  is >> str;
  is.get();  // consume the space.
  if (is.fail()) {
    KALDI_ERR << "Failed to read marker [started at file position "
              << pos_at_start << "], expected " << marker;
  }
  if (strcmp(str.c_str(), marker) != 0) {
    KALDI_ERR << "Expected marker \"" << marker << "\", got instead \""
              << str <<"\".";
  }
}

void ExpectMarker(std::istream &is, bool binary, const std::string &marker) {
  ExpectMarker(is, binary, marker.c_str());
}

}  // end namespace kaldi
