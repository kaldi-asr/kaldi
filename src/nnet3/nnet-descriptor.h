// nnet3/nnet-descriptor.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_DESCRIPTOR_H_
#define KALDI_NNET3_NNET_DESCRIPTOR_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet3 {

/**
   \file nnet-descriptor.h

   This file contains class definitions for classes ForwardingDescriptor,
   SumDescriptor and Descriptor.  Basically this is code that specifies how
   we glue together the outputs of possibly several other network-nodes, as the
   input of a particular network node (or as an output of the network).  In the
   neural-network code we refer to the top-level descriptor which is
   Descriptor.  The InputDescriptor is a concatenation of features; each part
   is a SumDescriptor.  The SumDescriptor is a summation over a set of features
   of all the same dimension, each of which is represented by a
   ForwardingDescriptor.  A ForwardingDescriptor in the simplest case just
   takes just points you to a particular network node, but in general can do
   things like adding time offsets, and selecting different rows of its matrix
   from different inputs.  Unlike the other descriptors, a ForwardingDescriptor
   is in general a bit like a parse tree, in that it can in general contain
   other ForwardingDescriptors.

   The following gives an overview of the expressions that can appear in
   descriptors.  Caution; this is a simplification that overgenerates
   descriptors: not all combinations are allowed.
\verbatim
<descriptor>  ::=   <node-name>      ;; node name of kInput or kComponent node.
<descriptor>  ::=   Append(<descriptor>, <descriptor> [, <descriptor> ... ] )
<descriptor>  ::=   Sum(<descriptor>, <descriptor>)
<descriptor>  ::=   Const(<value>, <dimension>)    ;; e.g. Const(1.0, 512)
<descriptor>  ::=   Scale(<scale>, <descriptor>)   ;; e.g. Scale(-1.0, tdnn2)
;; Failover or IfDefined might be useful for time t=-1 in a RNN, for instance.
<descriptor>  ::=   Failover(<descriptor>, <descriptor>)   ;; 1st arg if computable, else 2nd
<descriptor>  ::=   IfDefined(<descriptor>)     ;; the arg if defined, else zero.
<descriptor>  ::=   Offset(<descriptor>, <t-offset> [, <x-offset> ] ) ;; offsets are integers
;; Switch(...) is intended to be used in clockwork RNNs or similar schemes.  It chooses
;; one argument based on the value of t (in the requested Index) modulo the number of
;; arguments
<descriptor>  ::=   Switch(<descriptor>, <descriptor> [, <descriptor> ...])
;; For use in clockwork RNNs or similar, Round() rounds the time-index t of the
;; requested Index to the next-lowest multiple of the integer <t-modulus>,
;; and evaluates the input argument for the resulting Index.
<descriptor>  ::=   Round(<descriptor>, <t-modulus>)  ;; <t-modulus> is an integer
;; ReplaceIndex replaces some <variable-name> (t or x) in the requested Index
;; with a fixed integer <value>.  E.g. might be useful when incorporating
;; iVectors; iVector would always have time-index t=0.
<descriptor>  ::=   ReplaceIndex(<descriptor>, <variable-name>, <value>)
\endverbatim

 */



/// A ForwardingDescriptor describes how we copy data from another NetworkNode,
/// or from multiple other NetworkNodes, possibly with a scalar weight.  In the
/// base case this can just be equivalent to giving the name of another
/// NetworkNode, but we also support things like time-offsets, selecting
/// depending on the index from multiple different inputs, and things like that.
///
/// Note: nodes of type kOutput (i.e. output nodes of the network) cannot appear
/// as inputs in any descriptor.  This is to simplify compilation.
class ForwardingDescriptor {
 public:
  // Given an Index that's requested at the output of this descriptor, maps it
  // to a (node_index, Index) pair that says where we are to get the data from.
  //
  virtual Cindex MapToInput(const Index &output) const = 0;

  // Return the feature dimension.
  virtual int32 Dim(const Nnet &nnet) const = 0;

  virtual ForwardingDescriptor *Copy() const = 0;

  /// This function is for use in things like clockwork RNNs, where shifting the
  /// time of the inputs and outputs of the network by some multiple integer n
  /// would leave things the same, but shifting by non-multiples would change the
  /// network structure.  It returns the smallest modulus to which this
  /// descriptor is invariant; the lowest common multiple of all descriptors in
  /// the network gives you the modulus for the whole network.
  virtual int32 Modulus() const { return 1; }

  // Write to string that will be one line of a config-file-like format.  The
  // opposite of Parse.
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const = 0;

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const = 0;

  /// This function returns the scale on the node-index 'node_index' when it
  /// appears in expressions inside this descriptor, or +infinity if it does not
  /// appear.  E.g. if the descriptor is just `Scale(tdnn2, 2.0)` and the node
  /// index for `tdnn2` is 4, then GetScaleForNode(4) would return 2.0.  If a
  /// particular node_index > 0 appears in different sub-expressions of the
  /// descriptor with different scales it is an error (it's not supported) and
  /// this function would crash.
  virtual BaseFloat GetScaleForNode(int32 node_index) const = 0;

  virtual ~ForwardingDescriptor() { }
  ForwardingDescriptor() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(ForwardingDescriptor);
};

/// SimpleForwardingDescriptor is the base-case of ForwardingDescriptor,
/// consisting of a source node in the graph with a given scalar weight (which
/// will in the normal case be 1.0).  The string representation in the
/// normal (scale=1.0) case is just the node-name, like `tdnn2`; if
/// the weight is not 1.0 it's something like `Scale(2.0, tdnn2)`
class SimpleForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &index) const;
  virtual int32 Dim(const Nnet &nnet) const;
  virtual ForwardingDescriptor *Copy() const;
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // Write to string that will be one line of a config-file-like format.  The
  // opposite of Parse.
  // written form is just the node-name of src_node_.
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;

  SimpleForwardingDescriptor(int32 src_node,
                             BaseFloat scale = 1.0):
      src_node_(src_node), scale_(scale) {
    KALDI_ASSERT(src_node >= 0);
  }
  virtual ~SimpleForwardingDescriptor() { }
 private:
  int32 src_node_;  // index of the source NetworkNode.
  BaseFloat scale_;  // Scale of the node in the expression; this will be 1.0
                     // unless you used a Scale(...) expression in your
                     // Descriptor.
};

/// Offsets in 't' and 'x' values of other ForwardingDescriptors.
/// Written form is:
///   `Offset(<descriptor>, <t-offset> [, <x-offset> ] )`
/// e.g. `Offset(tdnn2, -2)`
class OffsetForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) const;
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;

  // written form is: Offset(<src-written-form>, t-offset [, x-offset])
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;

  virtual int32 Modulus() const { return src_->Modulus(); }

  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // takes ownership of src.
  OffsetForwardingDescriptor(ForwardingDescriptor *src,
                             Index offset): src_(src), offset_(offset) { }

  virtual ~OffsetForwardingDescriptor() { delete src_; }


  // this function is not in the shared interface. it's used
  // in class ModelCollapser.
  const ForwardingDescriptor &Src() const { return *src_; }
 private:
  ForwardingDescriptor *src_;  // Owned here.
  Index offset_;  // The index-offset to be added to the index.
};

/// Chooses from different inputs based on the the time index modulo
/// (the number of ForwardingDescriptors given as inputs).  This is rarely
/// if ever used.  Written form is:
///  `Switch(<descriptor>, <descriptor> [, <descriptor> ...])`
/// e.g. `Switch(tdnn2a, tdnn2b, tdnn2c)`
class SwitchingForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) const;
  virtual int32 Dim(const Nnet &nnet) const { return src_[0]->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "Switch(<written-form-of-src1>, <written-form-of-src2>, ... )"
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names) const;

  virtual int32 Modulus() const;

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // takes ownership of items in src.
  SwitchingForwardingDescriptor(std::vector<ForwardingDescriptor*> &src):
      src_(src) { }
  virtual ~SwitchingForwardingDescriptor() { DeletePointers(&src_); }
 private:
  // Pointers are owned here.
  std::vector<ForwardingDescriptor*> src_;
};



/// For use in clockwork RNNs and the like, this forwarding-descriptor
/// rounds the time-index t down to the the closest t' <= t that is
/// an exact multiple of t_modulus_.
/// Written form is: `Round(<descriptor>, <t-modulus>)`
/// e.g.: `Round(tdnn2, 3)`
class RoundingForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) const;
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "Round(<written-form-of-src>, <t_modulus>)"
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names) const;

  virtual int32 Modulus() const { return t_modulus_; }

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // takes ownership of src.
  RoundingForwardingDescriptor(ForwardingDescriptor *src,
                               int32 t_modulus):
      src_(src), t_modulus_(t_modulus) { }

  virtual ~RoundingForwardingDescriptor() { delete src_; }
 private:
  ForwardingDescriptor *src_;
  int32 t_modulus_;
};

/// This ForwardingDescriptor modifies the indexes (n, t, x) by replacing one
/// of them (normally t) with a constant value and keeping the rest.
/// Written form is: `ReplaceIndex(<descriptor>, <variable-name>, <value>)`
/// e.g. `ReplaceIndex(ivector, t, 0)`
class ReplaceIndexForwardingDescriptor: public ForwardingDescriptor {
 public:
  enum VariableName { kN = 0, kT = 1, kX = 2};

  virtual Cindex MapToInput(const Index &ind) const;
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "ReplaceIndex(<written-form-of-src>, <variable-name>, <value>)"
  // where <variable-name> is either "t" or "x".
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names) const;

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // takes ownership of src.
  ReplaceIndexForwardingDescriptor(ForwardingDescriptor *src,
                                   VariableName variable_name,
                                   int32 value):
      src_(src), variable_name_(variable_name), value_(value) { }

  virtual ~ReplaceIndexForwardingDescriptor() { delete src_; }
 private:
  ForwardingDescriptor *src_;
  VariableName variable_name_;
  int32 value_;
};


/// Forward declaration.  This is declared in nnet-computation-graph.h.
class CindexSet;

/// This is an abstract base-class.  In the normal case a SumDescriptor is a sum
/// over one or more ForwardingDescriptors (all of which must be of the same
/// dimension).  However, it also allows for logic for dealing with cases where
/// only some terms in the sum are present, and only some are included in the
/// sum: for example, not just expressions like A + B but also A + (B if
/// present), or (A if present; if not, B).  It also handles
/// expressions involving adding a constant, e.g.
/// `Sum(Scale(tdnn2, -1.0), Const(1.0, 512))` (see ConstantSumDescriptor).
class SumDescriptor {
 public:
  /// Given an Index at the output of this Descriptor, append to "dependencies"
  /// a list of Cindexes that describes what inputs we potentially depend on.
  /// The output list is not necessarily sorted, and this function doesn't make
  /// sure that it's unique.
  virtual void GetDependencies(const Index &ind,
                               std::vector<Cindex> *dependencies) const = 0;

  /// This function exists to enable us to manage optional dependencies,
  /// i.e. for making sense of expressions like (A + (B is present)) and (A if
  /// present; if not, B).  Suppose we are trying to compute the index "ind",
  /// and the user represents that "cindex_set" is the set of Cindexes are
  /// available to the computation; then this function will return true if we
  /// can compute the expression given these inputs; and if so, will output to
  /// "used_inputs" the list of Cindexes that this expression will be a
  /// summation over.
  ///
  ///  @param [in] ind  The index that we want to compute at the output of the
  ///                   Descriptor.
  ///  @param [in] cindex_set  The set of Cindexes that are available at the
  ///                   input of the Descriptor.
  ///  @param [out] used_inputs If non-NULL, if this function returns true then
  ///                  to this vector will be *appended* the inputs that will
  ///                  actually participate in the computation.  Else (if non-NULL) it
  ///                  will be left unchanged.
  ///  @return Returns true if this output is computable given the provided
  ///          inputs.
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *used_inputs) const = 0;

  virtual int32 Dim(const Nnet &nnet) const = 0;

  virtual SumDescriptor *Copy() const = 0;

  virtual ~SumDescriptor() { }

  /// This function appends to "node_indexes" a list (not necessarily sorted or
  /// unique) of all the node indexes that this descriptor may forward data from.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const = 0;

  /// This function returns the scale on the node-index 'node_index' when it
  /// appears in expressions inside this descriptor.  E.g. if the descriptor is
  /// just `Scale(tdnn2, 2.0)` and the node index for `tdnn2` is 4, then
  /// GetScaleForNode(4) would return 2.0.  It will return +infinity if the node
  /// is >= 0 and does not appear in this descriptor.  If node_index < 0, it
  /// returns the constant offset value from this descriptor, which will equal
  /// 0.0 if there is no expression like `Const(1.0, 512)` in this node.  If a
  /// particular node_index > 0 appears in different sub-expressions of the
  /// descriptor with different scales it is an error (it's not supported) and
  /// this function would crash.
  virtual BaseFloat GetScaleForNode(int32 node_index) const = 0;

  // see Modulus function of ForwardingDescriptor for explanation.
  virtual int32 Modulus() const = 0;

  /// Write in config-file format.  Conventional Read and Write methods are not
  /// supported.
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const = 0;
};

/// This is the case of class SumDescriptor, in which we contain just one term,
/// and that term is optional (an IfDefined() expression).  That term is a
/// general SumDescriptor.
///  The written form is: `IfDefined(<descriptor>)`, e.g.
///   `IfDefined(Offset(lstm2.s, -3))`
class OptionalSumDescriptor: public SumDescriptor {
 public:
  virtual void GetDependencies(const Index &ind,
                               std::vector<Cindex> *dependencies) const;
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *used_inputs) const {
      return src_->IsComputable(ind, cindex_set, used_inputs) || true;
  }

  virtual int32 Dim(const Nnet &nnet) const;

  // This function appends to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;
  virtual int32 Modulus() const { return src_->Modulus(); }
  /// written form is: if required_ == true, "<written-form-of-src>"
  /// else "IfDefined(<written-form-of-src>)".
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;

  OptionalSumDescriptor(SumDescriptor *src): src_(src) { }
  virtual ~OptionalSumDescriptor() { delete src_; }
 private:
  SumDescriptor *src_;
};

/// This is the normal base-case of SumDescriptor which just wraps a
/// ForwardingDescriptor.  The written form is any valid ForwardingDescriptor,
/// e.g. in the simplest case just `tdnn3`.
/// See also ConstantSumDescriptor().
class SimpleSumDescriptor: public SumDescriptor {
 public:
  virtual void GetDependencies(const Index &ind,
                               std::vector<Cindex> *dependencies) const;
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *used_inputs) const;
  virtual int32 Dim(const Nnet &nnet) const;

  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // This function appends to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual int32 Modulus() const { return src_->Modulus(); }
  /// written form is: if required_ == true, "<written-form-of-src>"
  /// else "IfDefined(<written-form-of-src>)".
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;

  SimpleSumDescriptor(ForwardingDescriptor *src): src_(src) { }
  virtual ~SimpleSumDescriptor() { delete src_; }

  // this function is not in the shared interface. it's used
  // in class ModelCollapser.
  const ForwardingDescriptor &Src() const { return *src_; }
 private:
  ForwardingDescriptor *src_;
};


/// This is an alternative base-case of SumDescriptor (an alternative to
/// SimpleSumDescriptor) which represents a constant term, e.g. `Const(1.0,
/// 512)`.  Note that this is not allowed to appear inside conditionals
/// such as IfDefined() or Failover(); this is enforced in the parsing
/// code involving class GeneralDescriptor.
/// The written form is: `Const(<value>, <dimension>)`, e.g.
/// `Const(-1.0, 512)`
class ConstantSumDescriptor: public SumDescriptor {
 public:
  virtual void GetDependencies(const Index &ind,
                               std::vector<Cindex> *dependencies) const { }
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *used_inputs) const {
    return true;
  }
  virtual int32 Dim(const Nnet &nnet) const { return dim_; }
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const { }
  virtual int32 Modulus() const { return 1; }
  /// The written form is: `Const(<value>, <dimension>)`, e.g.
  /// `Const(-1.0, 512)`
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;

  ConstantSumDescriptor(BaseFloat value, int32 dim);
  virtual ~ConstantSumDescriptor() {}
 private:
  BaseFloat value_;
  int32 dim_;
};

/// BinarySumDescriptor can represent either A + B, or (A if defined, else B).
/// Other expressions such as A + (B if defined, else zero), (A if defined, else
/// zero) + (B if defined, else zero), and (A if defined, else B if defined,
/// else zero) can be expressed using combinations of the two provided options
/// for BinarySumDescriptor and the variant
class BinarySumDescriptor: public SumDescriptor {
 public:
  enum Operation {
    kSumOperation,  // A + B
    kFailoverOperation, // A if defined, else B.
  };
  virtual void GetDependencies(const Index &ind,
                               std::vector<Cindex> *dependencies) const;
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *used_inputs) const;
  virtual int32 Dim(const Nnet &nnet) const;
  virtual BaseFloat GetScaleForNode(int32 node_index) const;

  // This function appends to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  virtual void GetNodeDependencies(std::vector<int32> *node_indexes) const;
  virtual int32 Modulus() const;
  /// Written form is: if op_ == kSum then "Sum(<src1>, <src2>)";
  /// if op_ == kFailover, then "Failover(<src1>, <src2>)"
  /// If you need more than binary operations, just use Sum(a, Sum(b, c)).
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;
  BinarySumDescriptor(Operation op, SumDescriptor *src1, SumDescriptor *src2):
      op_(op), src1_(src1), src2_(src2) {}
  virtual ~BinarySumDescriptor() { delete src1_; delete src2_; }
 private:
  Operation op_;
  SumDescriptor *src1_;
  SumDescriptor *src2_;
};


// A Descriptor concatenates over its parts, so its feature-dimension will
// be the sum of the feature-dimensions of its parts.  In a valid Descriptor,
// "parts" will be nonempty.  Each part may be (in general) a summation, but
// usually a summation with just one term.
class Descriptor {
 public:
  int32 Dim(const Nnet &nnet) const;

  // The Parse method is used for reading a config-file-style represenation.
  // Internally this uses class GeneralDescriptor to read and normalize the
  // input.  Assumes the input has already been tokenized into an array of
  // strings by DescriptorTokenize(); it moves the begin-pointer "next_token" to
  // account for each token that it consumes.  Prints warning and returns false on
  // error (including if there was junk after the last token).  The input tokens
  // should be terminated with a token that says "end of input".
  bool Parse(const std::vector<std::string> &node_names,
             const std::string **next_token);

  // Write in config-file format.
  // if parts_.size() == 1, written form is just "<written-form-of-part0>"
  // otherwise, written form is "Append(<written-form-of-part0>, <written-form-of-part1>,  ... )".
  void WriteConfig(std::ostream &os,
                   const std::vector<std::string> &node_names) const;

  /// This function exists to enable us to manage optional dependencies,
  /// i.e. for making sense of expressions like (A + (B is present)) and (A if
  /// present; if not, B).  Suppose we are trying to compute the index "ind",
  /// and the user represents that "cindex_set" is the set of Cindexes are
  /// available to the computation; then this function will return true if we
  /// can compute the expression given these inputs; and if so, will output to
  /// "used_inputs" the list of Cindexes (not necessarily unique) that this
  /// expression will include.  Otherwise it will return false and set
  /// used_inputs to the empty vector.
  ///
  ///  @param [in] ind  The index that we want to compute at the output of the
  ///                   Descriptor.
  ///  @param [in] cindex_set  The set of Cindexes that are available at the
  ///                   input of the Descriptor.
  ///  @param [out] used_inputs If non-NULL, if this function returns true then
  ///                  to this vector will be *appended* the inputs that will
  ///                  actually participate in the computation.  Else (if non-NULL) it
  ///                  will be left unchanged.
  ///  @return Returns true if this output is computable given the provided
  ///          inputs.
  void GetDependencies(const Index &index,
                       std::vector<Cindex> *used_inputs) const;

  /// Has the same purpose and interface as the IsComputable function of the
  /// SumDescriptor function.   Outputs to used_inputs rather than appending
  /// to it, though.  used_inputs will not be sorted or have repeats removed.
  bool IsComputable(const Index &ind,
                    const CindexSet &cindex_set,
                    std::vector<Cindex> *used_inputs) const;

  // This function outputs to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  void GetNodeDependencies(std::vector<int32> *node_indexes) const;

  // see Modulus function of ForwardingDescriptor for explanation.
  int32 Modulus() const;

  /// Returns the number of parts that are concatenated over.
  int32 NumParts() const { return parts_.size(); }
  /// returns the n'th part.
  const SumDescriptor &Part(int32 n) const;

  Descriptor() { }
  /// Copy constructor
  Descriptor(const Descriptor &other) { *this = other; }
  /// Assignment operator.
  Descriptor &operator = (const Descriptor &other);
  /// Takes ownership of pointers in "parts".
  Descriptor(const std::vector<SumDescriptor*> &parts):
      parts_(parts) { }
  /// Destructor
  ~Descriptor() { Destroy(); }
 private:
  void Destroy();
  // the elements of parts_ are owned here.
  std::vector<SumDescriptor*> parts_;
};


/**
   This class is only used when parsing Descriptors.  It is useful for normalizing
   descriptors that are structured in an invalid or redundant way, into a
   form that can be turned into a real Descriptor.
 */
struct GeneralDescriptor {
  enum DescriptorType { kAppend, kSum, kFailover, kIfDefined, kOffset, kSwitch,
                        kRound, kReplaceIndex, kScale, kConst, kNodeName };

  // The Parse method is used for reading a config-file-style represenation.
  // Assumes the input has already been tokenized into an array of strings, and
  // it moves the begin-pointer "next_token" to account for token that it
  // consumes.  Calls KALDI_ERR on error.  The list of tokens should be
  // terminated with a string saying "end of input".  Does not check that all
  // the input has been consumed-- the caller should do that [check that
  // **next_token == "end of input" after calling.]
  static GeneralDescriptor *Parse(const std::vector<std::string> &node_names,
                                  const std::string **next_token);

  explicit GeneralDescriptor(DescriptorType t, int32 value1 = -1,
                             int32 value2 = -1, BaseFloat alpha = 0.0):
      descriptor_type_(t), value1_(value1), value2_(value2),
      alpha_(alpha) { }


  ~GeneralDescriptor() { DeletePointers(&descriptors_); }

  GeneralDescriptor *GetNormalizedDescriptor() const;

  Descriptor *ConvertToDescriptor();

  // prints in text form-- this is really only used for debug.
  void Print(const std::vector<std::string> &node_names,
             std::ostream &os);

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(GeneralDescriptor);

  DescriptorType descriptor_type_;

  // value1_ is only relevant if:
  //    (a) descriptor_type_ == kReplaceIndex (value1_ is 1 for t, 2 for x)
  //    (b) descriptor_type_ == kNodeName (value1_ is the index of the node)
  //    (c) descriptor_type_ == kOffset (value1_ is the t offset).
  //    (d) descriptor_type_ == kConst (value1_ is the dimension and alpha_
  //                                   is the value).
  int32 value1_;
  // value2_ is only relevant if
  //    (a) descriptor_type == kReplaceIndex (value2_ is the value
  //                                          we replace the index with).
  //    (b) descriptor_type_ == kOffset (value2_ is the x offset)
  int32 value2_;

  // alpha is only relevant if
  //  (a) descriptor_type == kScale, and this will be the scaling factor.
  //  (b) descriptor_type == kConst; this is the value, and value1_ is set to the
  //          dimension.
  BaseFloat alpha_;

  // For any descriptor types that take args of type kDescriptor, a list of those
  // args.  Pointers owned here.
  std::vector<GeneralDescriptor*> descriptors_;

  //  parses an Append() or Sum() or Switch() expression after the "Append(" or
  //  "Sum(" or "Switch(" has been read.
  void ParseAppendOrSumOrSwitch(const std::vector<std::string> &node_names,
                                const std::string **next_token);
  // parse an IfDefined() expression after the IfDefined( has already been
  // read.
  void ParseIfDefined(const std::vector<std::string> &node_names,
                      const std::string **next_token);
  // ... and so on.
  void ParseOffset(const std::vector<std::string> &node_names,
                   const std::string **next_token);
  void ParseSwitch(const std::vector<std::string> &node_names,
                   const std::string **next_token);
  void ParseFailover(const std::vector<std::string> &node_names,
                     const std::string **next_token);
  void ParseRound(const std::vector<std::string> &node_names,
                  const std::string **next_token);
  void ParseScale(const std::vector<std::string> &node_names,
                  const std::string **next_token);
  void ParseConst(const std::vector<std::string> &node_names,
                  const std::string **next_token);
  void ParseReplaceIndex(const std::vector<std::string> &node_names,
                         const std::string **next_token);



  // Used inside NormalizeAppend().  Return the number of terms there
  // would be in a single consolidated Append() expressions, and asserts that in
  // whichever branch of any other expressions we take, the number of terms is
  // the same.
  int32 NumAppendTerms() const;
  // Used inside NormalizeAppend().  Gets one of the appended terms from this
  // descriptor, with 0 <= term < NumAppendTerms().  Answer is newly allocated.
  GeneralDescriptor *GetAppendTerm(int32 term) const;


  // Normalizes w.r.t. Append expressions by moving Append() to the outside.
  // Called only at the top level.
  GeneralDescriptor *NormalizeAppend() const;

  // This call does all other types of normalization except for normalizing
  // Append() expressions (which is assumed to have been done already).  Returns
  // true if anything was changed.
  static bool Normalize(GeneralDescriptor *ptr);

  SumDescriptor *ConvertToSumDescriptor() const;
  ForwardingDescriptor *ConvertToForwardingDescriptor() const;

};




} // namespace nnet3
} // namespace kaldi

#endif
