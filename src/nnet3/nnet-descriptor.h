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
   SumDescriptor and InputDescriptor.  Basically this is code that specifies how
   we glue together the outputs of possibly several other network-nodes, as the
   input of a particular network node (or as an output of the network).  In the
   neural-network code we refer to the top-level descriptor which is
   InputDescriptor.  The InputDescriptor is a concatenation features; each part
   is a SumDescriptor.  The SumDescriptor is a summation over a set of features
   of all the same dimension, each of which is represented by a
   ForwardingDescriptor.  A ForwardingDescriptor in the simplest case just
   takes just points you to a particular network node, but in general can do
   things like adding time offsets, and selecting different rows of its matrix
   from different inputs.  Unlike the other descriptors, a ForwardingDescriptor
   is in general a bit like a parse tree, in that it can in general contain
   other ForwardingDescriptors.
 */



// A ForwardingDescriptor describes how we copy data from another NetworkNode,
// or from multiple other NetworkNodes.  In the simplest case this can just be
// equivalent to giving the name of another NetworkNode, but we also support
// things like time-offsets, selecting depending on the index from multiple
// different inputs, and things like that.
//
// note: nodes of type kOutput (i.e. output nodes of the network) cannot appear
// as inputs in any descriptor.  This is to simplify compilation.
class ForwardingDescriptor {
 public:
  // Given an Index that's requested at the output of this descriptor, maps it
  // to a (node_index, Index) pair that says where we are to get the data from.
  virtual Cindex MapToInput(const Index &output) const;

  // Return the feature dimension.
  virtual int32 Dim(const Nnet &nnet) const;
  
  virtual ForwardingDescriptor *Copy() const;

  // The Parse method is used for reading a config-file-style represenation.
  // Assumes the input has already been tokenized into an array of strings, and
  // it moves the begin-pointer "next_token" to account for token that it
  // consumes.  Calls KALDI_ERR on error.
  // The list of tokens should be terminated with a string saying "end of input".
  static ForwardingDescriptor *Parse(const std::vector<std::string> &node_names,
                                     const std::string **next_token);
  

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
                           const std::vector<std::string> &node_names) const;

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;
  
  virtual ~ForwardingDescriptor();
  ForwardingDescriptor() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(ForwardingDescriptor);
};

// SimpleForwardingDescriptor is the base-case of ForwardingDescriptor,
class SimpleForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) { return Cindex(src_node_, ind); }
  virtual int32 Dim(const Nnet &nnet) const;
  virtual ForwardingDescriptor *Copy() const;
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;

  
  // Write to string that will be one line of a config-file-like format.  The
  // opposite of Parse.
  // written form is just the node-name of src_node_.
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names);

  SimpleForwardingDescriptor(int32 src_node): src_node_(src_node) {
    KALDI_ASSERT(src_node >= 0);
  }
  virtual ~SimpleForwardingDescriptor();
 private:
  int32 src_node_;  // index of the source NetworkNode.
};

class OffsetForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) {
    Cindex answer = src_->MapToInput(ind);
    answer.second = answer.second + offset_;
    return answer;
  }
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;

  // written form is: Offset(<src-written-form>, t-offset [, x-offset])
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names);
  
  virtual int32 Modulus() const { return src_->Modulus(); }
  
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;
  
  // takes ownership of src.
  OffsetForwardingDescriptor(ForwardingDescriptor *src,
                             Index offset): src_(src), offset_(offset) { }
  
  virtual ~OffsetForwardingDescriptor();
 private:
  ForwardingDescriptor *src_;  // Owned here.
  Index offset_;  // The index-offset to be added to the index.
};

// Chooses from different inputs based on the the time index modulo
// (the number of ForwardingDescriptors given as inputs).
class SwitchingForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) {
    KALDI_ASSERT(!src_.empty());
    int32 size = src_.size(), mod = ind.t % size;
    // next line gets "mathematical" modulus, not broken "C" modulus.
    if (mod < 0) mod += size;
    return src_[mod]->MapToInput(ind);
  }
  virtual int32 Dim(const Nnet &nnet) const { return src_[0]->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "Switch(<written-form-of-src1>, <written-form-of-src2>, ... )"
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names);

  virtual int32 Modulus() const;
  
  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;

  // takes ownership of items in src.
  SwitchingForwardingDescriptor(std::vector<ForwardingDescriptor*> &src):
      src_(src) { }
  virtual ~SwitchingForwardingDescriptor();
 private:
  // Pointers are owned here.
  std::vector<ForwardingDescriptor*> src_; 
};



/// For use in clockwork RNNs and the like, this forwarding-descriptor
/// rounds the time-index t down to the the closest t' <= t that is
/// an exact multiple of t_modulus_.
class RoundingForwardingDescriptor: public ForwardingDescriptor {
 public:
  virtual Cindex MapToInput(const Index &ind) {
    KALDI_ASSERT(t_modulus_ >= 1);
    Cindex ans = src_->MapToInput(ind);
    int32 mod = ans.second.t % t_modulus_;
    if (mod < 0)
      mod += t_modulus_;
    ans.second.t -= mod;
    return ans;
  }
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "Round(<written-form-of-src>, <t_modulus>)"
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names) const;

  virtual int32 Modulus() const { return t_modulus_; }

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;

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
class ReplaceIndexForwardingDescriptor: public ForwardingDescriptor {  
 public:
  enum VariableName { kN, kT, kX };
  
  virtual Cindex MapToInput(const Index &ind) {
    Cindex ans = src_->MapToInput(ind);
    switch (variable_name_) {
      case kT: ans.second.t = value_; break;
      case kX: ans.second.x = value_; break;
      default:  // kN or any other value is not allowed (doesn't make sense
                // to change the minibatch index in this way).
        KALDI_ERR << "Invalid variable name";
    }    
    return ans;
  }
  virtual int32 Dim(const Nnet &nnet) const { return src_->Dim(nnet); }
  virtual ForwardingDescriptor *Copy() const;
  // Written form is "ReplaceIndex(<written-form-of-src>, <variable-name>, <value>)"
  // where <variable-name> is either "t" or "x".
  virtual void WriteConfig(std::ostream &os,
                          const std::vector<std::string> &node_names) const;

  /// This function appends to "node_indexes" all the node indexes
  // that this descriptor may access.
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;

  // takes ownership of src.
  ReplaceIndexForwardingDescriptor(VariableName variable_name,
                                   int32 value,
                                   ForwardingDescriptor *src):
      variable_name_(variable_name), value_(value), src_(src) { }
  
  virtual ~ReplaceIndexForwardingDescriptor() { delete src_; }
 private:
  VariableName variable_name_;
  int32 value_;

  ForwardingDescriptor *src_;
};


/// Forward declaration.  This is declared in nnet-computation-graph.h.
class CindexSet;

/// This is an abstract base-class.  In the normal case a SumDescriptor is a sum
/// over one or more terms, all each corresponding to a quantity of the same
/// dimension, each of which is a ForwardingDescriptor.  However, it also allows
/// for logic for dealing with cases where only some terms in the sum are
/// present, and only some are included in the sum: for example, not just
/// expressions like A + B but also A + (B if present), or (A if present; if not,
/// B).
class SumDescriptor {
 public:

  /// Given an Index at the output of this Descriptor, output a list of Cindexes
  /// that describes what inputs we potentially depend on.  The output list is
  /// not necessarily sorted, and this function doesn't make sure that it's unique,
  /// but it should be unique in allowed expressions, and we'll later be checking
  /// this in IsComputable().  [Basically, we forbid expressions like x + x within
  /// the sum, to avoid having to deal with coefficients].
  virtual void MapToInputs(const Index &ind,
                           std::vector<Cindex> *dependencies) const = 0;

  /// This function exists to enable us to manage optional dependencies,
  /// i.e. for making sense of expressions like (A + (B is present)) and (A if
  /// present; if not, B).  Suppose we are trying to compute the index "ind",
  /// and the user represents that "cindex_set" is the set of Cindexes are
  /// available to the computation; then this function will return true if we
  /// can compute the expression given these inputs; and if so, will output to
  /// "required_inputs" the set of Cindexes that this expression will be a
  /// summation over.  We ensure that this is unique by just dying if the same
  /// Cindex appears twice in the sum; this becomes a limitation on the kinds of
  /// expressions the user is allowed to create.
  ///
  ///  @param [in] ind  The index that we want to compute at the output of the
  ///                   Descriptor.
  ///  @param [in] cindex_set  The set of Cindexes that are available at the
  ///                   input of the Descriptor.
  ///  @param [out] required_inputs If non-NULL, if this function returns true
  ///                   the inputs that will actually participate in the
  ///                   computation are output to here.  Else (if non-NULL) it
  ///                   will be set to the empty vector.
  ///  @return Returns true if this output is computable given the provided
  ///          inputs.
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const;
  
  virtual int32 Dim(const Nnet &nnet) const = 0;

  virtual SumDescriptor *Copy() const = 0;
  
  virtual ~SumDescriptor() { }

  // This function appends to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const = 0;
  
  // see Modulus function of ForwardingDescriptor for explanation.
  virtual int32 Modulus() const = 0;

  // The Parse method is used for reading a config-file-style represenation.
  // Assumes the input has already been tokenized into an array of strings, and
  // it moves the begin-pointer "next_token" to account for token that it
  // consumes.  Calls KALDI_ERR on error.
  // The input tokens should be terminated with a token that says "end of input". 
  static SumDescriptor* Parse(const std::vector<std::string> &node_names,
                              const std::string **next_token);
  
  /// Write in config-file format.  Conventional Read and Write methods are not
  /// supported.
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names);


};

/// This is the simple case of class SumDescriptor, in which we
/// contain just one term (the term is a ForwardingDescriptor).
/// You can initialize with reqired = false in order to express
/// an optional quantity, like (A if defined, else zero).
class UnarySumDescriptor: public SumDescriptor {
 public:
  virtual void MapToInputs(const Index &ind,
                           std::vector<Cindex> *dependencies) const;
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const;
  virtual int32 Dim(const Nnet &nnet) const;
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;
  virtual int32 Modulus() const;
  /// written form is: if required_ == true, "<written-form-of-src>"
  /// else "IfDefined(<written-form-of-src>)".
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;
  
  UnarySumDescriptor(ForwardingDescriptor *src,
                     bool required = true):
      src_(src), required_(required) { }
 private:
  ForwardingDescriptor *src_;
  bool required_;
};


/// BinarySumDescriptor can represent either A + B, or (A if defined, else B).
/// Other expressions such as A + (B if defined, else zero), (A if defined, else
/// zero) + (B if defined, else zero), and (A if defined, else B if defined,
/// else zero) can be expressed using combinations of the two provided options
/// for BinarySumDescriptor and the variant
class BinarySumDescriptor: public SumDescriptor {
 public:
  enum Operation {
    kSum,  // A + B
    kFailover, // A if defined, else B.
  };
  virtual void MapToInputs(const Index &ind,
                           std::vector<Cindex> *dependencies) const;
  virtual bool IsComputable(const Index &ind,
                            const CindexSet &cindex_set,
                            std::vector<Cindex> *required_inputs) const;
  virtual int32 Dim(const Nnet &nnet) const;
  virtual void ComputeDependencies(std::vector<int32> *node_indexes) const;
  virtual int32 Modulus() const;
  /// Written form is: if op_ == kSum then "Sum(<src1>, <src2>)";
  /// if op_ == kFailover, then "Failover(<src1>, <src2>)"
  /// If you need more than binary operations, just use Sum(a, Sum(b, c)).
  virtual void WriteConfig(std::ostream &os,
                           const std::vector<std::string> &node_names) const;
  virtual SumDescriptor *Copy() const;
  BinarySumDescriptor(Operation op, SumDescriptor *src1, SumDescriptor *src2):
      op_(op), src1_(src1), src2_(src2) {}
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
  // Assumes the input has already been tokenized into an array of strings, and
  // it moves the begin-pointer "next_token" to account for token that it
  // consumes.  Prints warning and returns false on error (including if there
  // was junk after the last token).
  // The input tokens should be terminated with a token that says "end of input".
  bool Parse(const std::vector<std::string> &node_names,
             const std::string **next_token);
  
  // Write in config-file format.
  // if parts_.size() == 1, written form is just "<written-form-of-part0>"
  // otherwise, written form is "Append(<written-form-of-part0>, <written-form-of-part1>,  ... )".
  void WriteConfig(std::ostream &os,
                   const std::vector<std::string> &node_names);
  
  /// This function outputs [rather than appends] to "dependencies" all Cindexes
  /// that may be be used to to compute this index.  This list is not guaranteed
  /// unique, i.e. it may contain repeats.
  void MapToInputs(const Index &index,
                   std::vector<Cindex> *dependencies) const;

  /// Has the same purpose and interface as the IsComputable function of the
  /// SumDescriptor function.
  bool IsComputable(const Index &ind,
                    const CindexSet &cindex_set,                    
                    std::vector<Cindex> *required_inputs) const;
  
  // This function appends to "node_indexes" a list (not necessarily sorted or
  // unique) of all the node indexes that this descriptor may forward data from.
  void ComputeDependencies(std::vector<int32> *node_indexes) const {}

  // see Modulus function of ForwardingDescriptor for explanation.
  int32 Modulus() const;

  /// Returns the number of parts that are concatenated over.
  int32 NumParts() const { return parts_.size(); }
  /// returns the n'th part.
  const SumDescriptor &Part(int32 n) const;

  Descriptor() { }
  /// Copy constructor
  Descriptor(const Descriptor &other);
  /// Assignment operator.  
  Descriptor &operator = (const Descriptor &other);
  /// Destructor
  ~Descriptor() { Destroy(); }
 private:
  void Destroy(); // empties parts_ after deleting its members.
  // the elements of parts_ are owned here.
  std::vector<SumDescriptor*> parts_;
};




} // namespace nnet3
} // namespace kaldi

#endif
