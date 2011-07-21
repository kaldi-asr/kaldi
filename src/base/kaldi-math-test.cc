// base/kaldi-math-test.cc
// Copyright 2009-2011  Microsoft Corporation;  Yanmin Qian;  Jan Silovsky

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
#include "base/kaldi-math.h"

namespace kaldi {

template<class I> void UnitTestGcdTpl() {
  for (I a = 1; a < 15; a++) {  // a is min gcd.
    I b = (I)(rand() % 10);
    I c = (I)(rand() % 10);
    if (rand()%2 == 0 && std::numeric_limits<I>::is_signed) b = -b;
    if (rand()%2 == 0 && std::numeric_limits<I>::is_signed) c = -c;
    if (b == 0 && c == 0) continue;  // gcd not defined for such numbers.
    I g = Gcd(b*a, c*a);
    assert(g >= a);
    assert((b*a) % g == 0);
    assert((c*a) % g == 0);
  }
}

void UnitTestRoundUpToNearestPowerOfTwo() {
  assert(RoundUpToNearestPowerOfTwo(1) == 1);
  assert(RoundUpToNearestPowerOfTwo(2) == 2);
  assert(RoundUpToNearestPowerOfTwo(3) == 4);
  assert(RoundUpToNearestPowerOfTwo(4) == 4);
  assert(RoundUpToNearestPowerOfTwo(7) == 8);
  assert(RoundUpToNearestPowerOfTwo(8) == 8);
  assert(RoundUpToNearestPowerOfTwo(255) == 256);
  assert(RoundUpToNearestPowerOfTwo(256) == 256);
  assert(RoundUpToNearestPowerOfTwo(257) == 512);
  assert(RoundUpToNearestPowerOfTwo(1073700000) == 1073741824  );
}

void UnitTestGcd() {
  UnitTestGcdTpl<int>();
  UnitTestGcdTpl<char>();
  UnitTestGcdTpl<size_t>();
  UnitTestGcdTpl<unsigned short>();
}

void UnitTestRand() {
  // Testing random-number generation.
  using namespace kaldi;
  std::cout << "Testing random-number generation.  "
            << "If there is an error this may not terminate.\n";
  std::cout << "If this does not terminate, look more closely.  "
            << "There might be a problem [but might not be]\n";
  for (int i = 1; i < 10; i++) {
    {  // test RandUniform.
      std::cout << "Test RandUniform\n";
      assert(RandUniform() >= 0 && RandUniform() <= 1);
      float sum = RandUniform()-0.5;
      for (int j = 0; ; j++) {
        sum += RandUniform()-0.5;
        if (std::abs(sum) < 0.5*sqrt((double)j)) break;
      }
    }
    {  // test RandGauss.
      float sum = RandGauss();
      for (int j = 0; ; j++) {
        sum += RandGauss();
        if (std::abs(sum) < 0.5*sqrt((double)j)) break;
      }
    }
    {  // test poisson_rand().
      assert(RandPoisson(3.0) >= 0);
      assert(RandPoisson(0.0) == 0);
      std::cout << "Test RandPoisson\n";
      float lambda = RandUniform() * 3.0;  // between 0 and 3.
      double sum = RandPoisson(lambda) - lambda;  // expected value is zero.
      for (int j = 0; ; j++) {
        sum += RandPoisson(lambda) - lambda;
        if (std::abs(sum) < 0.5*sqrt((double)j)) break;
      }
    }

    {  // test RandInt().
      assert(RandInt(0, 3) >= 0 && RandInt(0, 3) <= 3);

      std::cout << "Test RandInt\n";
      int minint = rand() % 200;
      int maxint = minint + 1 + rand()  % 20;

      float sum = RandInt(minint, maxint) +  0.5*(minint+maxint);
      for (int j = 0; ; j++) {
        sum += RandInt(minint, maxint) - 0.5*(minint+maxint);
        if (std::abs((float)sum) < 0.5*sqrt((double)j)*(maxint-minint)) break;
      }
    }
  }
}

void UnitTestLogAddSub() {
  using namespace kaldi;
  for (int i = 0; i < 100; i++) {
    double f1 = rand() % 10000, f2 = rand() % 20;
    double add1 = exp(LogAdd(log(f1), log(f2)));
    double add2 = exp(LogAdd(log(f2), log(f1)));
    double add = f1 + f2, thresh = add*0.00001;
    assert(std::abs(add-add1) < thresh && std::abs(add-add2) < thresh);


    try {
      double f2_check = exp(LogSub(log(add), log(f1))), thresh = (f2*0.01)+0.001;
      assert(std::abs(f2_check-f2) < thresh);
    } catch(...) {
      assert(f2 == 0);  // It will probably crash for f2=0.
    }
  }
}

void UnitTestDefines() {  // Yes, we even unit-test the preprocessor statements.
  assert(exp(kLogZeroFloat) == 0.0);
  assert(exp(kLogZeroDouble) == 0.0);
  BaseFloat den = 0.0;
  assert(KALDI_ISNAN(0.0 / den));
  assert(!KALDI_ISINF(0.0 / den));
  assert(!KALDI_ISFINITE(0.0 / den));
  assert(!KALDI_ISNAN(1.0 / den));
  assert(KALDI_ISINF(1.0 / den));
  assert(!KALDI_ISFINITE(1.0 / den));
  assert(KALDI_ISFINITE(0.0));
  assert(!KALDI_ISINF(0.0));
  assert(!KALDI_ISNAN(0.0));

  std::cout << 1.0+DBL_EPSILON;
  std::cout << 1.0 + 0.5*DBL_EPSILON;
  assert(1.0 + DBL_EPSILON != 1.0 && 1.0 + (0.5*DBL_EPSILON) == 1.0
         && "If this test fails, you can probably just comment it out-- may mean your CPU exceeds expected floating point precision");
  assert(1.0f + FLT_EPSILON != 1.0f && 1.0f + (0.5f*FLT_EPSILON) == 1.0f
         && "If this test fails, you can probably just comment it out-- may mean your CPU exceeds expected floating point precision");
  assert(std::abs(sin(M_PI)) < 1.0e-05 && std::abs(cos(M_PI)+1.0) < 1.0e-05);
  assert(std::abs(sin(M_2PI)) < 1.0e-05 && std::abs(cos(M_2PI)-1.0) < 1.0e-05);
  assert(std::abs(sin(exp(M_LOG_2PI))) < 1.0e-05);
  assert(std::abs(cos(exp(M_LOG_2PI)) - 1.0) < 1.0e-05);
}

void UnitTestAssertFunc() {  // Testing Assert*** functions
  using namespace kaldi;
  for (int i = 1; i < 100; i++) {
    float f1 = rand() % 10000 + 1, f2 = rand() % 20 + 1;
    float tmp1 = f1 * f2;
    float tmp2 = (1/f1 + 1/f2);
    float tmp3 = (1/(f1 - 1.0) + 1/(f2 - 1.0));
    float tmp4 = (1/(f1 + 1.0) + 1/(f2 + 1.0));
    float add = f1 + f2;
    float addeql = tmp1 * tmp2, addgeq = tmp1 * tmp3, addleq = tmp1 * tmp4;
    float thresh = 0.00001;
    AssertEqual(add, addeql, thresh);  // test AssertEqual()
    AssertGeq(addgeq, add, thresh);  // test AsserGeq()
    AssertLeq(addleq, add, thresh);  // test AsserLeq()
  }
}

template<class I> void UnitTestFactorizeTpl() {
  for (int p= 0; p < 100; p++) {
    I m = rand() % 100000;
    if (m >= 1) {
      std::vector<I> factors;
      Factorize(m, &factors);
      I m2 = 1;
      for (size_t i = 0; i < factors.size(); i++) {
        m2 *= factors[i];
        if (i+1 < factors.size())
          assert(factors[i+1] >= factors[i]);  // check sorted.
      }
      assert(m2 == m);  // check correctness.
    }
  }
}

void UnitTestFactorize() {
  UnitTestFactorizeTpl<int>();
  UnitTestFactorizeTpl<size_t>();
  UnitTestFactorizeTpl<unsigned short>();
}

void UnitTestApproxEqual() {
  KALDI_ASSERT(ApproxEqual(1.0, 1.00001));
  KALDI_ASSERT(ApproxEqual(1.0, 1.00001, 0.001));
  KALDI_ASSERT(!ApproxEqual(1.0, 1.1));
  KALDI_ASSERT(!ApproxEqual(1.0, 1.01, 0.001));
  KALDI_ASSERT(!ApproxEqual(1.0, 0.0));
  KALDI_ASSERT(ApproxEqual(0.0, 0.0));
  KALDI_ASSERT(!ApproxEqual(0.0, 0.00001));
  KALDI_ASSERT(!ApproxEqual(std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(ApproxEqual(std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(ApproxEqual(-std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(!ApproxEqual(-std::numeric_limits<float>::infinity(),
                            0));
  KALDI_ASSERT(!ApproxEqual(-std::numeric_limits<float>::infinity(),
                            1));
               
}

}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestApproxEqual();
  UnitTestGcd();
  UnitTestFactorize();
  UnitTestDefines();
  UnitTestLogAddSub();
  UnitTestRand();
  UnitTestAssertFunc();
  UnitTestRoundUpToNearestPowerOfTwo();
}

