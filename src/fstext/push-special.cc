// fstext/push-special.cc

// Copyright 2012  Johns Hopkins University (authors: Daniel Povey,
//                 Ehsan Variani, Pegah Ghahrmani)

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

#include "fstext/push-special.h"
#include "base/kaldi-error.h"

namespace fst {

/*
  In pushing algorithm:
    Each state gets a potential, say p(s).  Initial state must have potential zero.
    The states have transition probabilities to each other, call these w(s, t), and
    final-probabilities f(s).  One special state is the initial state.
    These are real probabilities, think of them like HMM transition probabilities; we'll
    represent them as double.  Each state has a potential, p(s).
    After taking into account the potentials, the weights transform
       w(s, t) -->  w(s, t) / p(s) * p(t)
    and the final-probs transform
       f(s)  -->  f(s) / p(s).
    The initial state's potential is fixed at 1.0.
    Let us define a kind of normalizer for each state s as:
       n(s) = f(s) + \sum_t w(s, t),
    or taking into account the potentials, and treating the self-loop as special,
       
       n(s) =  f(s)/p(s) + \sum_t w(s, t) p(t) / p(s)
            =  w(s, s) + (1/p(s)) f(s) + \sum_{t != s} w(s, t) p(t).         (Eq. 1)
     
    This should be one if the FST is stochastic (in the log semiring).
    In fact not all FSTs can be made stochastic while preserving equivalence, so
    in "PushSpecial" we settle for a different condition: that all the n(s) be the
    same.  This means that the non-sum-to-one-ness of the FST is somehow smeared
    out throughout the FST.  We want an algorithm that makes all the n(s) the same,
    and we formulate it in terms of iteratively improving objective function.  The
    objective function will be the sum-of-squared deviation of each of the n(s) from
    their overall mean, i.e.
       \sum_s  (n(s) - n_{avg})^2
    where n_avg is the average of the n(s).  When we choose an s to optimize its p(s),
    we'll minimize this function, but while minimizing it we'll treat n_{avg} as
    a fixed quantitiy.  We can show that even though this is not 100% accurate, we
    still end up minimizing the objective function (i.e. we still decrease the
    sum-of-squared difference).
    
    Suppose we choose some s for which to tweak p(s). [naturally s cannot be the start
    state].  Firstly, we assume n_{avg} is a known and fixed quantity.  When we
    change p(s) we alter n(s) and also n(t) for all states t != s that have a transition
    into s (i.e. w(s, t) != 0).  Let's write p(s) for the current value of p(s),
    and p'(s) for the value we'll replace it with, and use similar notation for
    the n(s).  We'll write out the part of the objective function that involves p(s),
    and this is:

     F =  (n(s) - n_{avg})^2  +  \sum_{t != s}  (n(t) - n_{avg})^2.

    Here, n_{avg} is treated as fixed.  We can write n(s) as:
       n(s) = w(s, s) + k(s) / p(s)
    where k(s) = f(s) + \sum_{t != s) w(s, t) p(t),
    but note that if we have n(s) already, k(s) can be computed by:
         k(s) = (n(s) - w(s, s)) * p(s)

    We can write n(t) [for t != s] as:
       n(t) = j(t) + w(t, s)/p(t) p(s)
    and
       j(t) = w(t, t) + (1/p(t)) \sum_{u != s, u != t} w(t, u) p(u)
    but in practice if we have the normalizers n(t) up to date,
    we can compute it more efficiently as
       j(t) = n(t) - w(t, s)/p(t) p(s)                      (Eq. 2)
       

    Now let's imagine we did the substitutions for n(s) and n(t), and we'll
    write out the terms in F that are functions of p(s).  We have:

       F =                         k(s)^2  p(s)^{-2}
             + 2 k(s) (w(s, s) -  n_{avg}) p(s)^{-1}
             + [constant term that doesn't matter]
+ (\sum_t 2(j(t) - n_{avg})(w(t, s)/p(t))  p(s)
             + (\sum_t  (w(t, s)/p(t))^2 ) p(s)^2                 (Eq. 3)

    Note that the {-2} and {+2} power terms are both positive, and this means
    that F will get large as p(s) either gets too large or too small.  This is
    comforting because we want to minimize F.  Let us write the four coefficients
    above as c_{-2}, c_{-1}, c_1 and c_2.   The minimum of F can be found where
    the derivative of F w.r.t. p(s) is zero.  Here, let's just call it p for short.
    This will be where:
     d/dp  c_{-2} p^{-2} + c_{-1} p^{-1} + c_1 p + c_2 p^2  = 0
            -2 c_{-2} p^{-3} - c_{-1} p^{-2} + c_1 + c_2 p  = 0 .
    Technically we can solve this type of formula by means of the quartic equations,
    but these take up pages and pages.  Instead we'll use a one-dimensional form
    of Newton's method, computing the derivative and second derivative by differentiating the
    formula.

*/
            
class PushSpecialClass {
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

 public:
  // Everything happens in the initializer.
  PushSpecialClass(VectorFst<StdArc> *fst,
                   float delta): fst_(fst) {
    num_states_ = fst_->NumStates();
    initial_state_ = fst_->Start();
    occ_.resize(num_states_, 1.0 / sqrt(num_states_)); // unit length
    
    pred_.resize(num_states_);
    for (StateId s = 0; s < num_states_; s++) {
      for (ArcIterator<VectorFst<StdArc> > aiter(*fst, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        StateId t = arc.nextstate;
        double weight = exp(-arc.weight.Value());
        pred_[t].push_back(std::make_pair(s, weight));
      }
      double final = exp(-fst_->Final(s).Value());
      if (final != 0.0)
        pred_[initial_state_].push_back(std::make_pair(s, final));
    }
    Iterate(delta);
    ModifyFst();
  }
 private:
  double TestAccuracy() { // returns the error (the difference
    // between the min and max weights).
    double min_sum = 0, max_sum = 0;
    for (StateId s = 0; s < num_states_; s++) {
      double sum = 0.0;
      for (ArcIterator<VectorFst<StdArc> > aiter(*fst_, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        StateId t = arc.nextstate;
        sum += exp(-arc.weight.Value()) * occ_[t] / occ_[s];
      }
      sum += exp(-(fst_->Final(s).Value())) * occ_[initial_state_] / occ_[s];
      if (s == 0) {
        min_sum = sum;
        max_sum = sum;
      } else {
        min_sum = std::min(min_sum, sum);
        max_sum = std::max(max_sum, sum);
      }
    }
    KALDI_VLOG(4) << "min,max is " << min_sum << " " << max_sum;
    return log(max_sum / min_sum); // In FST world we'll actually
    // dealing with logs, so the log of the ratio is more suitable
    // to compare with delta (makes testing the algorithm easier).
  }

  
  void Iterate(float delta) {
    // This is like the power method to find the top eigenvalue of a matrix.
    // We limit it to 2000 iters max, just in case something unanticipated
    // happens, but we should exit due to the "delta" thing, usually after
    // several tens of iterations.
    int iter;
    for (iter = 0; iter < 2000; iter++) {
      std::vector<double> new_occ(num_states_);
      // We initialize new_occ to 0.1 * occ.  A simpler algorithm would
      // initialize them to zero, so it's like the pure power method.  This is
      // like the power method on (M + 0.1 I), and we do it this way to avoid a
      // problem we encountered with certain very simple linear FSTs where the
      // eigenvalues of the weight matrix (including negative and imaginary
      // ones) all have the same magnitude.
      for (int i = 0; i < num_states_; i++)
        new_occ[i] = 0.1 * occ_[i];
      
      for (int i = 0; i < num_states_; i++) {
        std::vector<std::pair<StateId, double> >::const_iterator iter,
            end = pred_[i].end();
        for (iter = pred_[i].begin(); iter != end; ++iter) {
          StateId j = iter->first;
          double p = iter->second;
          new_occ[j] += occ_[i] * p;
        }
      }
      double sumsq = 0.0;
      for (int i = 0; i < num_states_; i++) sumsq += new_occ[i] * new_occ[i];
      lambda_ = std::sqrt(sumsq);
      double inv_lambda = 1.0 / lambda_;
      for (int i = 0; i < num_states_; i++) occ_[i] = new_occ[i] * inv_lambda;
      KALDI_VLOG(4) << "Lambda is " << lambda_;
      if (iter % 5 == 0 && iter > 0 && TestAccuracy() <= delta) {
        KALDI_VLOG(3) << "Weight-pushing converged after " << iter
                      << " iterations.";
        return;
      }
    }
    KALDI_WARN << "push-special: finished " << iter
               << " iterations without converging.  Output will be inaccurate.";
  }
  
 
  // Modifies the FST weights and the final-prob to take account of these potentials.
  void ModifyFst() {
    // First get the potentials as negative-logs, like the values
    // in the FST.
    for (StateId s = 0; s < num_states_; s++) {
      occ_[s] = -log(occ_[s]);
      if (KALDI_ISNAN(occ_[s]) || KALDI_ISINF(occ_[s]))
        KALDI_WARN << "NaN or inf found: " << occ_[s];
    }
    for (StateId s = 0; s < num_states_; s++) {
      for (MutableArcIterator<VectorFst<StdArc> > aiter(fst_, s);
           !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        StateId t = arc.nextstate;
        arc.weight = Weight(arc.weight.Value() + occ_[t] - occ_[s]);
        aiter.SetValue(arc);
      }
      fst_->SetFinal(s, Times(fst_->Final(s).Value(),
                              Weight(occ_[initial_state_] - occ_[s])));
    }
  }

 private:
  StateId num_states_;
  StateId initial_state_;
  std::vector<double> occ_; // the top eigenvector of (matrix of weights) transposed.
  double lambda_; // our current estimate of the top eigenvalue.
  
  std::vector<std::vector<std::pair<StateId, double> > > pred_; // List of transitions
  // into each state.  For the start state, this list consists of the list of
  // states with final-probs, each with their final prob.
  
  VectorFst<StdArc> *fst_;
  
};




void PushSpecial(VectorFst<StdArc> *fst, float delta) {
  if (fst->NumStates() > 0)
    PushSpecialClass c(fst, delta); // all the work
  // gets done in the initializer.
}

  
} // end namespace fst.


/*
  Note: in testing an earlier, simpler
  version of this method (without the 0.1 * old_occ) we had a problem with the following FST.
0	2	3	3	0
1	3	1	4	0.5
2	1	0	0	0.5
3	0.25

 Corresponds to the following matrix [or maybe its transpose, doesn't matter
 probably]

 a=exp(-0.5)
 b=exp(-0.25)
M =   [ 0 1 0 0
       0 0 a 0
       0 0 0 a
       b 0 0 0 ]

eigs(M)
eigs(M)

ans =

  -0.0000 - 0.7316i
  -0.0000 + 0.7316i
   0.7316
  -0.7316

   OK, so the issue appears to be that all the eigenvalues of this matrix
   have the same magnitude.  The solution is to work with the eigenvalues
   of M + alpha I, for some small alpha such as 0.1 (so as not to slow down
   convergence in the normal case).

*/
