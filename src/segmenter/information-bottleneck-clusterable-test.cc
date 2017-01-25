
#include "base/kaldi-common.h"
#include "segmenter/information-bottleneck-clusterable.h"

namespace kaldi {

static void TestClusterable() {
  {
    Vector<BaseFloat> a_vec(3);
    a_vec(0) = 0.5;
    a_vec(1) = 0.5;
    int32 a_count = 100;
    KALDI_ASSERT(ApproxEqual(a_vec.Sum(), 1.0));

    Vector<BaseFloat> b_vec(3);
    b_vec(1) = 0.333;
    b_vec(2) = 0.667;
    int32 b_count = 100;
    KALDI_ASSERT(ApproxEqual(b_vec.Sum(), 1.0));

    InformationBottleneckClusterable a(1, a_count, a_vec);
    InformationBottleneckClusterable b(2, b_count, b_vec);
  
    Vector<BaseFloat> sum_vec(a_vec.Dim());
    sum_vec.AddVec(a_count, a_vec);
    sum_vec.AddVec(b_count, b_vec);
    sum_vec.Scale(1.0 / (a_count + b_count));
    KALDI_ASSERT(ApproxEqual(sum_vec.Sum(), 1.0));

    InformationBottleneckClusterable sum(3);
    InformationBottleneckClusterable c(3);

    sum.Add(a);
    sum.Add(b);

    c.AddStats(1, a_count, a_vec);
    c.AddStats(2, b_count, b_vec);

    KALDI_ASSERT(c.Counts() == sum.Counts());
    KALDI_ASSERT(ApproxEqual(c.Objf(), sum.Objf()));
    KALDI_ASSERT(ApproxEqual(-c.Objf() + a.Objf() + b.Objf(), a.Distance(b)));
    KALDI_ASSERT(sum_vec.ApproxEqual(c.RelevanceDist()));
    KALDI_ASSERT(sum_vec.ApproxEqual(sum.RelevanceDist()));
  }
 
  for (int32 i = 0; i < 100; i++) {
    int32 dim = RandInt(2, 10);
    
    Vector<BaseFloat> a_vec(dim);
    a_vec.SetRandn();
    a_vec.ApplyPowAbs(1.0);
    a_vec.Scale(1 / a_vec.Sum());
    KALDI_ASSERT(ApproxEqual(a_vec.Sum(), 1.0));
    int32 a_count = RandInt(1, 100);
    InformationBottleneckClusterable a(1, a_count, a_vec);
    
    Vector<BaseFloat> b_vec(dim);
    b_vec.SetRandn();
    b_vec.ApplyPowAbs(1.0);
    b_vec.Scale(1 / b_vec.Sum());
    KALDI_ASSERT(ApproxEqual(b_vec.Sum(), 1.0));
    int32 b_count = RandInt(1, 100);
    InformationBottleneckClusterable b(2, b_count, b_vec);
    
    Vector<BaseFloat> sum_vec(a_vec.Dim());
    sum_vec.AddVec(a_count, a_vec);
    sum_vec.AddVec(b_count, b_vec);
    sum_vec.Scale(1.0 / (a_count + b_count));
    KALDI_ASSERT(ApproxEqual(sum_vec.Sum(), 1.0));
    
    InformationBottleneckClusterable sum(dim);
    InformationBottleneckClusterable c(dim);

    sum.Add(a);
    sum.Add(b);

    c.AddStats(1, a_count, a_vec);
    c.AddStats(2, b_count, b_vec);

    KALDI_ASSERT(c.Counts() == sum.Counts());
    KALDI_ASSERT(ApproxEqual(c.Objf(), sum.Objf()));
    KALDI_ASSERT(ApproxEqual(-c.Objf() + a.Objf() + b.Objf(), a.Distance(b)));
    KALDI_ASSERT(sum_vec.ApproxEqual(c.RelevanceDist()));
    KALDI_ASSERT(sum_vec.ApproxEqual(sum.RelevanceDist()));
  }
}

}  // end namespace kaldi
 
int main() {
  using namespace kaldi;

  TestClusterable();
}
