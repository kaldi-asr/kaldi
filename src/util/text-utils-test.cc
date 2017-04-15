// util/text-utils-test.cc

// Copyright 2009-2011     Microsoft Corporation
//                2017     Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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
  return static_cast<char>(32 + Rand() % 95);  // between ' ' and '~'
}

const char *ws_delim = " \t\n\r";
char GetRandDelim() {
  if (Rand() % 2 == 0)
    return static_cast<char>(33 + Rand() % 94);  // between '!' and '~';
  else
    return ws_delim[Rand() % 4];
}


void TestSplitStringToVector() {
  // srand((unsigned int)time(NULL));
  // didn't compile on cygwin.

  {
    std::vector<std::string> str_vec;
    SplitStringToVector("", " ", false, &str_vec);
    KALDI_ASSERT(str_vec.size() == 1);  // If this fails it may just mean
    // that someone changed the
    // semantics of SplitStringToVector in a reasonable way.
    SplitStringToVector("", " ", true, &str_vec);
    KALDI_ASSERT(str_vec.empty());
  }
  for (int j = 0; j < 100; j++) {
    std::vector<std::string> str_vec;
    int sz = Rand() % 73;
    std::string full;
    for (int i = 0; i < sz-1; i++) {
      full.push_back((Rand() % 7 == 0)? GetRandDelim() : GetRandChar());
    }
    std::string delim;
    delim.push_back(GetRandDelim());
    bool omit_empty_strings = (Rand() %2 == 0)? true : false;
    SplitStringToVector(full, delim.c_str(), omit_empty_strings, &str_vec);
    std::string new_full;
    for (size_t i = 0; i < str_vec.size(); i++) {
      if (omit_empty_strings) KALDI_ASSERT(str_vec[i] != "");
      new_full.append(str_vec[i]);
      if (i < str_vec.size() -1) new_full.append(delim);
    }
    std::string new_full2;
    JoinVectorToString(str_vec, delim.c_str(), omit_empty_strings, &new_full2);
    if (omit_empty_strings) {  // sequences of delimiters cannot be matched
      size_t start = full.find_first_not_of(delim),
          end = full.find_last_not_of(delim);
      if (start == std::string::npos) {  // only delimiters
        KALDI_ASSERT(end == std::string::npos);
      } else {
        std::string full_test;
        char last = '\0';
        for (size_t i = start; i <= end; i++) {
          if (full[i] != last || last != *delim.c_str())
            full_test.push_back(full[i]);
          last = full[i];
        }
        if (!full.empty()) {
          KALDI_ASSERT(new_full.compare(full_test) == 0);
          KALDI_ASSERT(new_full2.compare(full_test) == 0);
        }
      }
    } else if (!full.empty()) {
      KALDI_ASSERT(new_full.compare(full) == 0);
      KALDI_ASSERT(new_full2.compare(full) == 0);
    }
  }
}

void TestSplitStringToIntegers() {
  {
    std::vector<int32> v;
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4:", ":", false, &v) == false);
    KALDI_ASSERT(SplitStringToIntegers(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers("-1\n2\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    KALDI_ASSERT(SplitStringToIntegers("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<uint32> v;
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4", ":", false, &v) == false);
    // cannot put negative number in uint32.
  }
}



void TestSplitStringToFloats() {
  {
    std::vector<float> v;
    KALDI_ASSERT(SplitStringToFloats("-1:2.5:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats("-1:2.5:4:", ":", false, &v) == false);
    KALDI_ASSERT(SplitStringToFloats(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats("-1\n2.5\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    KALDI_ASSERT(SplitStringToFloats("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<double> v;
    KALDI_ASSERT(SplitStringToFloats("-1:2:4", ":", false, &v) == true);
  }
}

void TestConvertStringToInteger() {
  int32 i;
  KALDI_ASSERT(ConvertStringToInteger("12345", &i) && i == 12345);
  KALDI_ASSERT(ConvertStringToInteger("-12345", &i) && i == -12345);
  char j;
  KALDI_ASSERT(!ConvertStringToInteger("-12345", &j));  // too big for char.

  KALDI_ASSERT(ConvertStringToInteger(" -12345 ", &i));  // whitespace accepted

  KALDI_ASSERT(!ConvertStringToInteger("a ", &i));  // non-integers rejected.

  KALDI_ASSERT(ConvertStringToInteger("0", &i) && i == 0);

  uint64 k;
  KALDI_ASSERT(ConvertStringToInteger("12345", &k) && k == 12345);
  KALDI_ASSERT(!ConvertStringToInteger("-12345", &k));  // unsigned,
                                                        // cannot convert.
}

template<class Real>
void TestConvertStringToReal() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal("1", &d) && d == 1.0);
  KALDI_ASSERT(ConvertStringToReal("-1", &d) && d == -1.0);
  KALDI_ASSERT(ConvertStringToReal("-1", &d) && d == -1.0);
  KALDI_ASSERT(ConvertStringToReal(" -1 ", &d) && d == -1.0);
  KALDI_ASSERT(!ConvertStringToReal("-1 x", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1f", &d));
  KALDI_ASSERT(ConvertStringToReal("12345.2", &d) && fabs(d-12345.2) < 1.0);
  KALDI_ASSERT(ConvertStringToReal("1.0e+08", &d) && fabs(d-1.0e+08) < 100.0);

  // it also works for inf or nan.
  KALDI_ASSERT(ConvertStringToReal("inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("inf ", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" inf ", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("+inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-inf", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("Inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("INF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("InF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("infinity", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-infinity", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE inf", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEinf", &d));
  KALDI_ASSERT(!ConvertStringToReal("infGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("inf_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("inf GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE infinity", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEinfinity", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinityGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinity_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinity GARBAGE", &d));
  KALDI_ASSERT(ConvertStringToReal("1.#INF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-1.#INF", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-1.#INF  ", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" -1.#INF ", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE 1.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE1.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("2.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("-2.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("1.#INFGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("1.#INF_GARBAGE", &d));

  KALDI_ASSERT(ConvertStringToReal("nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("+nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("-nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("Nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NaN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" NaN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NaN ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" NaN ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("1.#QNAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("-1.#QNAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("1.#QNAN  ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" 1.#QNAN ", &d) && d != d);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE nan", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEnan", &d));
  KALDI_ASSERT(!ConvertStringToReal("nanGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("nan_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("nan GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE 1.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE1.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("2.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("-2.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1.#QNAN_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1.#QNANGARBAGE", &d));
}

template<class Real>
void TestNan() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal(std::to_string(sqrt(-1)), &d) && d != d);
}

template<class Real>
void TestInf() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal(std::to_string(exp(10000)), &d) &&
               d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(std::to_string(-exp(10000)), &d) &&
               d < 0 && d - d != 0);
}


std::string TrimTmp(std::string s) {
  Trim(&s);
  return s;
}

void TestTrim() {
  KALDI_ASSERT(TrimTmp(" a ") == "a");
  KALDI_ASSERT(TrimTmp(" a b  c") == "a b  c");
  KALDI_ASSERT(TrimTmp("") == "");
  KALDI_ASSERT(TrimTmp("X\n") == "X");
  KALDI_ASSERT(TrimTmp("X\n\t") == "X");
  KALDI_ASSERT(TrimTmp("\n\tX") == "X");
}  // end namespace kaldi


void TestSplitStringOnFirstSpace() {
  std::string a, b;
  SplitStringOnFirstSpace("a b", &a, &b);
  KALDI_ASSERT(a == "a" && b == "b");
  SplitStringOnFirstSpace("aa bb", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb");
  SplitStringOnFirstSpace("aa", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "");
  SplitStringOnFirstSpace(" aa \n\t ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "");
  SplitStringOnFirstSpace("  \n\t ", &a, &b);
  KALDI_ASSERT(a == "" && b == "");
  SplitStringOnFirstSpace(" aa   bb \n\t ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb");
  SplitStringOnFirstSpace(" aa   bb cc ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb cc");
  SplitStringOnFirstSpace(" aa   bb  cc ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb  cc");
  SplitStringOnFirstSpace(" aa   bb  cc", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb  cc");
}

void TestIsToken() {
  KALDI_ASSERT(IsToken("a"));
  KALDI_ASSERT(IsToken("ab"));
  KALDI_ASSERT(!IsToken("ab "));
  KALDI_ASSERT(!IsToken(" ab"));
  KALDI_ASSERT(!IsToken("a b"));
  KALDI_ASSERT(IsToken("\231"));  // typical non-ASCII printable character,
                                  // something with an accent.
  KALDI_ASSERT(!IsToken("\377"));  // character 255, which is a form of space.
  KALDI_ASSERT(IsToken("a-b,c,d=ef"));
  KALDI_ASSERT(!IsToken("a\nb"));
  KALDI_ASSERT(!IsToken("a\tb"));
  KALDI_ASSERT(!IsToken("ab\t"));
  KALDI_ASSERT(!IsToken(""));
}

void TestIsLine() {
  KALDI_ASSERT(IsLine("a"));
  KALDI_ASSERT(IsLine("a b"));
  KALDI_ASSERT(!IsLine("a\nb"));
  KALDI_ASSERT(!IsLine("a b "));
  KALDI_ASSERT(!IsLine(" a b"));
}


void TestStringsApproxEqual() {
  // we must test the test.
  KALDI_ASSERT(!StringsApproxEqual("a", "b"));
  KALDI_ASSERT(!StringsApproxEqual("1", "2"));
  KALDI_ASSERT(StringsApproxEqual("1.234", "1.235", 2));
  KALDI_ASSERT(!StringsApproxEqual("1.234", "1.235", 3));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y", "x 1.2345 y", 3));
  KALDI_ASSERT(!StringsApproxEqual("x 1.234 y", "x 1.2345 y", 4));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 1));
  KALDI_ASSERT(!StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 2));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.411", 2));
  KALDI_ASSERT(StringsApproxEqual("x 1.0 y", "x 1.0001 y", 3));
  KALDI_ASSERT(!StringsApproxEqual("x 1.0 y", "x 1.0001 y", 4));
}


}  // end namespace kaldi

int main() {
  using namespace kaldi;
  TestSplitStringToVector();
  TestSplitStringToIntegers();
  TestSplitStringToFloats();
  TestConvertStringToInteger();
  TestConvertStringToReal<float>();
  TestConvertStringToReal<double>();
  TestTrim();
  TestSplitStringOnFirstSpace();
  TestIsToken();
  TestIsLine();
  TestStringsApproxEqual();
  TestNan<float>();
  TestNan<double>();
  TestInf<float>();
  TestInf<double>();
  std::cout << "Test OK\n";
}
