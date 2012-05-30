// util/text-utils-test.cc

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


#include "base/kaldi-common.h"
#include "util/text-utils.h"

namespace kaldi {

char GetRandChar() {
  return static_cast<char>(32 + rand() % 95);  // between ' ' and '~'
}

const char* ws_delim = " \t\n\r";
char GetRandDelim() {
  if (rand() % 2 == 0)
    return static_cast<char>(33 + rand() % 94);  // between '!' and '~';
  else
    return ws_delim[rand() % 4];
}


void TestSplitStringToVector() {
  // srand((unsigned int)time(NULL));
  // didn't compile on cygwin.

  {
    std::vector<std::string> str_vec;
    SplitStringToVector("", " ", &str_vec, false);
    assert(str_vec.size() == 1);  // If this fails it may just mean
    // that someone changed the
    // semantics of SplitStringToVector in a reasonable way.
    SplitStringToVector("", " ", &str_vec, true);
    assert(str_vec.empty());
  }
  for (int j = 0; j < 100; j++) {
    std::vector<std::string> str_vec;
    int sz = rand() % 73;
    std::string full;
    for (int i = 0; i < sz-1; i++) {
      full.push_back( (rand() % 7 == 0)? GetRandDelim() : GetRandChar());
    }
    std::string delim;
    delim.push_back(GetRandDelim());
    bool omit_empty_strings = (rand() %2 == 0)? true : false;
    SplitStringToVector(full, delim.c_str(), &str_vec, omit_empty_strings);
    std::string new_full;
    for (size_t i = 0; i < str_vec.size(); i++) {
      if (omit_empty_strings) assert(str_vec[i] != "");
      new_full.append(str_vec[i]);
      if (i < str_vec.size() -1) new_full.append(delim);
    }
    if (omit_empty_strings) {  // sequences of delimiters cannot be matched
      size_t start = full.find_first_not_of(delim),
          end = full.find_last_not_of(delim);
      if (start == std::string::npos) {  // only delimiters
        assert(end == std::string::npos);
      } else {
        std::string full_test;
        char last = '\0';
        for (size_t i = start; i <= end; i++) {
          if (full[i] != last || last != *delim.c_str())
            full_test.push_back(full[i]);
          last = full[i];
        }
        if (!full.empty())
          assert(new_full.compare(full_test) == 0);
      }
    } else if (!full.empty())
      assert(new_full.compare(full) == 0);
  }
}

void TestSplitStringToIntegers() {
  {
    std::vector<int32> v;
    assert(SplitStringToIntegers("-1:2:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers("-1:2:4:", ":", false, &v) == false);
    assert(SplitStringToIntegers(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers("-1\n2\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    assert(SplitStringToIntegers(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    assert(SplitStringToIntegers("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<uint32> v;
    assert(SplitStringToIntegers("-1:2:4", ":", false, &v) == false);
    // cannot put negative number in uint32.
  }
}

void TestConvertStringToInteger() {
  int32 i;
  assert(ConvertStringToInteger("12345", &i) && i == 12345);
  assert(ConvertStringToInteger("-12345", &i) && i == -12345);
  char j;
  assert(!ConvertStringToInteger("-12345", &j));  // too big for char.

  assert(ConvertStringToInteger(" -12345 ", &i));  // whitespace accepted

  assert(!ConvertStringToInteger("a ", &i));  // non-integers rejected.

  assert(ConvertStringToInteger("0", &i) && i == 0);

  uint64 k;
  assert(ConvertStringToInteger("12345", &k) && k == 12345);
  assert(!ConvertStringToInteger("-12345", &k));  // unsigned, cannot convert.

}

template<class Real>
void TestConvertStringToReal() {
  Real d;
  assert(ConvertStringToReal("1", &d) && d == 1.0);
  assert(ConvertStringToReal("-1", &d) && d == -1.0);
  assert(ConvertStringToReal("-1", &d) && d == -1.0);
  assert(ConvertStringToReal(" -1 ", &d) && d == -1.0);
  assert(!ConvertStringToReal("-1 x", &d));
  assert(!ConvertStringToReal("-1f", &d));
  assert(ConvertStringToReal("12345.2", &d) && fabs(d-12345.2) < 1.0);
  assert(ConvertStringToReal("1.0e+08", &d) && fabs(d-1.0e+08) < 100.0);
}


std::string TrimTmp(std::string s) {
  Trim(&s);
  return s;
}

void TestTrim() {
  assert(TrimTmp(" a ") == "a");
  assert(TrimTmp(" a b  c") == "a b  c");
  assert(TrimTmp("") == "");
  assert(TrimTmp("X\n") == "X");
  assert(TrimTmp("X\n\t") == "X");
  assert(TrimTmp("\n\tX") == "X");
} // end namespace kaldi


void TestSplitStringOnFirstSpace() {
  std::string a, b;
  SplitStringOnFirstSpace("a b", &a, &b);
  assert(a == "a" && b == "b");
  SplitStringOnFirstSpace("aa bb", &a, &b);
  assert(a == "aa" && b == "bb");
  SplitStringOnFirstSpace("aa", &a, &b);
  assert(a == "aa" && b == "");
  SplitStringOnFirstSpace(" aa \n\t ", &a, &b);
  assert(a == "aa" && b == "");
  SplitStringOnFirstSpace("  \n\t ", &a, &b);
  assert(a == "" && b == "");
  SplitStringOnFirstSpace(" aa   bb \n\t ", &a, &b);
  assert(a == "aa" && b == "bb");
  SplitStringOnFirstSpace(" aa   bb cc ", &a, &b);
  assert(a == "aa" && b == "bb cc");
  SplitStringOnFirstSpace(" aa   bb  cc ", &a, &b);
  assert(a == "aa" && b == "bb  cc");
  SplitStringOnFirstSpace(" aa   bb  cc", &a, &b);
  assert(a == "aa" && b == "bb  cc");
}

void TestIsToken() {
  assert(IsToken("a"));
  assert(IsToken("ab"));
  assert(!IsToken("ab "));
  assert(!IsToken(" ab"));
  assert(!IsToken("a b"));
  assert(IsToken("\231")); // typical non-ASCII printable character, something with
  // an accent.
  assert(!IsToken("\377")); // character 255, which is a form of space.
  assert(IsToken("a-b,c,d=ef"));
  assert(!IsToken("a\nb"));
  assert(!IsToken("a\tb"));
  assert(!IsToken("ab\t"));
  assert(!IsToken(""));
}

void TestIsLine() {
  assert(IsLine("a"));
  assert(IsLine("a b"));
  assert(!IsLine("a\nb"));
  assert(!IsLine("a b "));
  assert(!IsLine(" a b"));
}

} // end namespace kaldi

int main() {
  using namespace kaldi;
  TestSplitStringToVector();
  TestSplitStringToIntegers();
  TestConvertStringToInteger();
  TestConvertStringToReal<float>();
  TestConvertStringToReal<double>();
  TestTrim();
  TestSplitStringOnFirstSpace();
  TestIsToken();
  TestIsLine();
  std::cout << "Test OK\n";
}



