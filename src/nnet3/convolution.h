// nnet3/convolution.h

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CONVOLUTION_H_
#define KALDI_NNET3_NNET_CONVOLUTION_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "nnet3/nnet-common.h"

#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  convolution.h
///
/// This file contains some fairly low-level utilities for implementing
/// convolutional neural networks and related methods such as TDNNs, which are
/// mostly used in nnet-convolutional-component.h.  This would not necessarily
/// be suitable as a general-purpose and self-contained setup for convolution,
/// as it is quite linked with the overall framework of the nnet3 library.
/// (the underlying ideas might be usable, though).
///
/// We have chosen to implement this here, rather than using CuDNN, because we
/// realized that it was quite easy to efficiently implement CNNs in the nnet3
/// framework in a way that would support both GPUs and CPUs, at least for the
/// typical setups that have small patch dimensions (like 1x1 or 3x3).  In a
/// typical 3x3 convolution, the entire convolution can be done using 3 matrix
/// multiplies (and 3 corresponding CopyColsFromMat calls).


namespace time_height_convolution {

/**
   This comment explains the basic framework used for everything related to
   time-height convolution.  We are doing convolution in 2 dimensions; these
   would normally be width and height, but in the nnet3 framework we identify
   the width with the 'time' dimension (the 't' element of an Index).  This
   enables us to use this framework in the normal way for speech tasks, and it
   turns out to have other advantages it too, giving us a very efficient and
   easy implementation of CNNs (basically, the nnet3 framework takes care of
   certain reorderings for us).  As mentioned, the 't' index will correspond to
   the width, and the vectors we operate on will be of dimension height *
   num-filters, where the filter-index has the stride of 1.

   We will use the GeneralComponent interface, and its function
   ReorderIndexes(), to ensure that the input and output Indexes of the
   component have a specified regular structure; we'll pad with 'blank' Indexes
   (t=kNoTime) on the input and output of the component, as needed to ensure
   that it's an evenly spaced grid over n and t, with x always zero and the t
   values evenly spaced.  (However, a note on even spacing: for computations
   with downsampling this ordering of the 't' values is bit more complicated,
   search for 'blocks' in the rest of this header for more information).

   First consider the simplest case, call it "same-t-stride" (where there is no
   downsampling on the time index, i.e.  the input and output 't' values have
   the same stride, like 1, 2 or 4).  The input and output matrices have
   dimension num-t-values * num-images, with the num-t-values having the higher
   stride.  The computation involves copying a row-range of the input matrix to
   a temporary matrix with a column mapping (the temporary matrix will typically
   have more columns than the input matrix); and then doing a matrix-multiply
   between the reshaped temporary matrix and a block of the parameters; the
   block corresponds to a particular time-offset.  Then we may need to repeat
   the whole process with a different, shifted row-range of the input matrix and
   a different column map.  You may have to read the rest of this header, to
   understand this in more detail.
 */


/**
   This struct represents a convolutional model from a structural point
   of view (it doesn't contain the actual parameters).  Note: the parameters
   are to be stored in a matrix of dimension (num_filters_out) by
   (offsets.size() * num_filters_in) [where the offset-index has the larger
   stride than the filter-index].

   Partly out of a desire for generality, but also for convenience in
   implementation and integration with nnet3, at this level we don't represent
   the patch size in the normal way like '1x1' or '3x3', but as a list of pairs
   (time-offset, height-offset).  E.g. a 1x1 patch would normally be the single
   pair (0,0), and a 3x3 patch might be represented as

   offsets={ (0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2) }

   However-- and you have to be a bit careful here-- the time indexes are on an
   *absolute* numbering scheme so that if you had downsampled the time axis on a
   previous layer, the time-offsets might all be multiples of (e.g.) 2 or 4, but
   the height-offsets would normally always be separated by 1.  [note: we always
   normalize the list of (time-offset, height-offset) pairs with the
   lexicographical ordering that you see above.]  This asymmetry between time
   and height may not be very aesthetic, but the absolute numbering of time is
   at the core of how the framework works.  Note: the offsets don't have to
   start from zero, they can be less than zero, just like the offsets in TDNNs
   which are often lists like (-3,0,3).  Don't be surprised to see things like:

   offsets={ (-3,-1),(-3,0),(-3,1), (0,-1),(0,0),(0,2), (3,-1),(3,0),(3,1) }

   If there are negative offsets in the height dimension (as above) it means
   that there is zero-padding in the height dimension (because the first
   height-index at both the input and the output is 0, so having a height-offset
   means that to compute the output at height-index  0 we need the input at
   height-index -1, which doesn't exist; this implies zero padding on the
   bottom of the image.
 */
struct ConvolutionModel {
  int32 num_filters_in;   // number of input filters, e.g. 128.
  int32 num_filters_out;  // number of output filters, e.g. 256.
  int32 height_in;   // image height in, e.g. 40.
  int32 height_out;  // image height out, e.g. 40 (no subsampling or zero
                     // padding), 38 (with zero padding) (or for an example with
                     // 2x subsampling and no zero-padding: maybe 20).
  int32 height_subsample_out;  // subsampling factor for height.  In the 3
                               // examples given for height_out above, would be
                               // 1, 1 and 2 respectively.
  struct Offset {
    int32 time_offset;
    int32 height_offset;
    // give it a lexicographic ordering.
    inline bool operator < (const Offset &other) const {
      if (time_offset < other.time_offset) return true;
      else if (time_offset > other.time_offset) return false;
      else return height_offset < other.height_offset;
    }
    inline bool operator <= (const Offset &other) const {
      if (time_offset < other.time_offset) return true;
      else if (time_offset > other.time_offset) return false;
      else return height_offset <= other.height_offset;
    }
    inline bool operator == (const Offset &other) const {
      return time_offset == other.time_offset &&
          height_offset == other.height_offset;
    }
  };
  // For a 3x3 patch, the 'offsets' vector would be a list of 9 elements.  It's
  // always unique and sorted in lexicographic order.  See the extended comment
  // for struct ConvolutionModel for an explanation.
  std::vector<Offset> offsets;

  // This set, 'required_time_offsets', relates to zero-padding on the time
  // axis.  It should consist of a nonempty subset of the time-offset values
  // that have been seen in offsets[*].time_offset.  If there is no zero-padding
  // on the time (width) axis it would be that entire set.  If there is
  // zero-padding it would in most circumstances contain just the middle one,
  // e.g. of {0,1,2} we'd keep just {1}, or of {-3,0,3} we'd keep just {0}.  The
  // way to understand it is that all the time-offsets define dependencies in
  // the computation, but the list of 'required' offsets determines when a
  // computation can proceed when some of the dependencies are not present (any
  // non-required depenencies that were not present default to zero).
  std::set<int32> required_time_offsets;

  // This variable, which is derived from 'offsets', stores all the time offsets
  // that are present there, i.e. all the values of 'offsets[*].time_offset'
  std::set<int32> all_time_offsets;

  // This variable, which is derived from 'offsets', is the greatest common
  // divisor of the differences between the members of 'all_time_offsets';
  // e.g. if 'all_time_offsets' is {1,3,5} it would be 2.  It is used to figure
  // out what grid structure the input to the computation should have.  It is
  // set to zero if all_time_offsets.size() == 1.
  int32 time_offsets_modulus;


  // Computes the derived parameters 'all_time_offsets' and
  // 'time_offsets_modulus'.
  void ComputeDerived();

  // You'll notice that there is nothing here that explicitly specifies the
  // padding.  At this level, any padding on the height axis is implicit.  For
  // example, suppose there is a height-offset of -1, that implies we must be
  // padding at the bottom by at least 1, because the output height-index starts
  // from 0, and it would require the input at height -1, whereas the input
  // height-index starts from 0.  All padding is implicitly zero-padding.
  // Padding in the height dimension depends on (height_in, height_out,
  // height_subsample_out) and the 'height_offset' members of 'offsets'; padding
  // in the time dimension depends on 'required_time_offset'
  // vs. 'all_time_offsets'.

  // the InputDim() and OutputDim() really relate to its behavior in a
  // neural-net component, they are the input-dim and output-dim of the features
  // that the component has as input/output; physically, this is the column
  // dimension at the input and output of the component.  The time dimension
  // corresponds to the row-index of those features.
  int32 InputDim() const { return num_filters_in * height_in; }
  int32 OutputDim() const { return num_filters_out * height_out; }
  // number of rows in the parameter matrix
  int32 ParamRows() const { return num_filters_out; }
  // number of cols in the parameter matrix
  int32 ParamCols() const { return num_filters_in * static_cast<int32>(offsets.size()); }

  ConvolutionModel() { }

  bool operator == (const ConvolutionModel &other) const;

  /*
    Checks that this model makes sense, and returns true if so; if not, returns
    false (and if it's for certain less-obvious reasons, prints a warning first
     explaining why)..

   @param [in] check_heights_used  If true, part of the check is that all
         height-values at the input are used at some point (if they
         are not, this model is probably not what you intended).
   @param [in] allow_height_padding  If true, the checking code assumes that
         zero-padding on the height axis is permitted.
   @return  Returns true if the check passed, false otherwise.
  */
  bool Check(bool check_heights_used = true,
             bool allow_height_padding = true) const;

  // Returns an info-string that describes the model; it looks like
  // "num-filters-in=32, num-filters-out=64, height-in=40, height-out=40, ... ".
  // It's suitable for use in the 'info' output of the convolutional component.
  std::string Info() const;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


/**
   This struct represents the structure of a convolution computation.
   This is used inside the PrecomputedIndexes object for
   the TimeHeightConvolutionComponent (it depends on the inputs and
   outputs as well as the layer).

   *CAUTION*: this is after certain transformations of the problem, so the
   height_in may not always be the "real" height of the input image (it may be a
   multiple thereof), and the num_t_in may not always be the "real" number
   of distinct time-steps on the input of the computation (it may be a divisor
   thereof).  ConvolutionComputation contains the info needed to actually
   perform the computation.
*/
struct ConvolutionComputation {
  // num_filters_in and num_filters_out will be the same as in the model.
  int32 num_filters_in, num_filters_out;
  // height_out will be the same as in the model, but height_in may be
  // affected by reshaping (may be larger than the model's height_in).
  int32 height_in, height_out;
  // num_t_in and num_t_out are the number of rows in the input and output
  // matrices, but num_t_in may be affected by reshaping (may be smaller
  // than the model's num_t_in).
  // num_t_in will be >= num_times_out, and if it's greater it will be greater by a
  // small additive term, not by a multiplicative factor.
  int32 num_t_in, num_t_out;
  // num_images is the number of (n,x) pairs present in the input/output
  // indexes (although in most setups the x values will all be zero and
  // they will only vary in n).
  int32 num_images;

  // temp_rows and temp_cols define the size of a temporary matrix that the
  // computation uses.  temp_rows is the number of rows in that temporary
  // matrix; it will normally be equal to [multiplying from greatest to least
  // stride], (num_times_out * num_images), but it may be less in order to save
  // memory.  The execution code is in charge of looping over the data using
  // this matrix, in order to ensure that we cover all output rows.  If you are
  // just trying to understand the framework, assume that it's always equal to
  // num_times_out * num_images.

  // Note: if all of the steps[*].columns_are_contiguous values are true AND all
  // of the steps[*].columns.Dim() equal the input-num-cols (=num_filters_in *
  // height_in), then the temporary matrix is never needed and in that case,
  // temp_rows and temp_cols will both be zero.
  int32 temp_rows, temp_cols;

  // There may be a few steps in the computation (e.g. in a 3x3 convolution
  // without subsampling, there would be 3 steps), and the output is a summation
  // over contributions from each step.  each step has a different value
  // 'input_time_shift' (which is the number of input rows to discard at the
  // start of the input matrix, and won't be the same as the increment in 't',
  // if t_step_in in the ConvolutionComputationIo != 1.
  struct ConvolutionStep {
    // input_time_shift >= 0 is the number of initial time-indexes of the input
    // (i.e. the number of initial rows of the matrix) that we discard for this
    // step. We may discard some final time-indexes too, if needed so that the
    // total number of input time-indexes equals the total number of output
    // time-indexes.
    int32 input_time_shift;

    // params_start_col >= 0 says the start-column-index of the parameter matrix
    // where we start a sub-matrix to be used in this step (the num-cols of that
    // sub-matrix is given by columns.Dim() / height_out).
    int32 params_start_col;

    // height_map is the 'upstream' parameter from which 'columns' and
    // 'backward_columns' are derived; it compactly defines a column mapping
    // that is used when copying the input to a temporary matrix.
    // height_map.dim() * num_filters_in gives the num-cols in this temporary
    // matrix.  Each element of 'height_map' corresponds to a column range of
    // 'num_filters_in' columnn of the temporary matrix, and it says which
    // (same-sized) column-range of the input matrix is to be used as the source
    // for this data.  Its elements are in the range -1 <= height_map[i] <
    // num_filters_in, where -1's are used for blocks that have zero values.
    // height_map would be the same as 'columns' if num_filters_in == 1.
    std::vector<int32> height_map;

    // 'columns' is derived from 'pixel_map'.
    // columns.Dim() <= temp_cols is the num-columns of
    // a sub-matrix of the temporary matrix, that we
    // populate on this step.
    //
    // -1 <= columns[i] < height_in * num_filters_in
    // gives the dimension of the (reshaped) input to copy
    // If columns[i] == -1, it means write a zero.
    CuArray<int32> columns;

    // 'backward_columns' is derived from 'columns', it is used in
    // the backprop.  Each element of 'backward_columns' has the
    // same dim as the num-cols of the input matrix.  It's basically
    // the reverse map of 'columns', but split into multiple parts (and
    // padded with -1's as necessary) so that we can process elements
    // of the input which are copied multiple times to the temporary
    // matrix.
    std::vector<CuArray<int32> > backward_columns;

    // 'columns_are_contiguous' is derived from 'columns'; it's true if
    // 'columns' is a contiguous range of nonnegative integers, like '20, 21,
    // 22, ... '.
    bool columns_are_contiguous;
    // 'first_column' is derived from 'columns'; it equals columns[0].  It is
    // only of interest if 'columns_are_contiguous' is true (it enables an
    // optimization).
    int32 first_column;
  };
  std::vector<ConvolutionStep> steps;


  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Computes derived variables in 'steps', i.e. 'columns', 'backward_columns',
  // columns_are_contiguous, and 'first_column'.
  void ComputeDerived();

  // check that this computation makes sense; crash if not.
  void Check() const;
};



/**
   This struct contains options for compiling the convolutional computation.
 */
struct ConvolutionComputationOptions {
  // max_memory_mb determines how many megabytes of memory we are willing to use
  // for the temporary matrix.  If it would exceed this amount, we do the
  // computation in batches.
  BaseFloat max_memory_mb;
  ConvolutionComputationOptions(): max_memory_mb(200.0) { }
};



// This struct represents the structure of the input and output of a
// convolutional computation (the input and output images; not the model itself,
// which is represented by ConvolutionModel).  We require that both the input
// and output indexes have a regular repeated structure, and if this is not the
// case then the input and output indexes will be padded with 'blank' indexes
// (indexes having a 't' vlaue of kNoTime) as needed to fit them into regular
// grids.  In addition 'blank' indexes may be added to reflect zero-padding on
// the input.
struct ConvolutionComputationIo {
  int32 num_images;  // 'num_images' is the number of distinct (n,x) values in
                     // the indexes.  Normally the x values would all be zero
                     // and the n values would go from 0 to num_images - 1, but
                     // this is not required.  We do enforce (via padding) that
                     // each (n,x) pair, i.e. each image, is associated with the
                     // same number of 't' values.

  // the following represents the set of 't' values on the input and output.
  // their meaning is obvious, but we should note that if there is just one
  // output or input index, we will set the step to zero when initially
  // creating this struct, and it may get set to other values later on, mostly
  // to avoid creating extra code paths.
  int32 start_t_in, t_step_in, num_t_in;
  int32 start_t_out, t_step_out, num_t_out;

  // reorder_t_in will be 1 in normal cases (no downsampling), but it may have values
  // greater than 1 (e.g. 2 if we're downsampling by a factor of 2).
  // This doesn't affect the set of indexes on the input, but it affects how they
  // are ordered.
  //
  //   If reorder_t_in == 1 then order the indexes one block for all
  // indexes with t=t0=start_t_in; then one block for all
  // t=t1=(start_t_in+t_step_in); then one block for t=t2, t=t3, and so on.
  //
  //   If reorder_t_in is >1 (for example, 2), then the values for t0 and t1 would
  // be interspersed in a single block; then the values for t1 and t2 would
  // be interspersed in the next block; and so on.  Within these blocks,
  // it's the 't' values that have the smaller stride.  This ordering allows
  // a reshaping such that we can imagine that the input and output have the
  // same 't' increment; it's useful in subsampling convolutions..
  int32 reorder_t_in;
};

/**
   Check that this model and this I/O request are compatible in
   terms of required context, etc, and crash if not.
   if allow_extra_input == false, this will crash if the
   input 'io' object has time values that would never be
   used because they are before/after the first/last
   desired time values. */
void CheckModelAndIo(const ConvolutionModel &model,
                     const ConvolutionComputationIo &io,
                     bool allow_extra_input = false);


/**
   This function does the compilation for a convolution computation; it's
   a wrapper for the functions below, which should not have to be called
   by the end user.

   @param [in] model  The convolution model that this computation is for.
   @param [in] input_indexes   The list of Indexes available at the input of
                      the computation.
   @param [in] output_indexes  The list of Indexes requested to be computed
                      at the output of the computation.  It is an error if
                      all dependencies are not satisfied (specifically: for
                      each Index (n,t,x) in 'output_indexes', the Index
                      (n,t+time_offset,x) must be present in 'input_indexes'
                      for each time_offset in model.required_time_offsets.
   @param [out] computation  If non-NULL, the compiled computation will be
                      written to this location.

 */
void CompileConvolutionComputation(
    const ConvolutionModel &model,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    const ConvolutionComputationOptions &opts,
    ConvolutionComputation *computation,
    std::vector<Index> *input_indexes_modified,
    std::vector<Index> *output_indexes_modified);


/**
   \brief This does the forward computation of convolution.  (note: this is
         convolution without a bias term; you have to handle that separately).

   @param [in] conv_comp  A struct that describes the computation
                          to be performed.
   @param [in] input     The input to the convolution.  This
             should be of dimension (or should be reshapable to
             the dimension) conv_comp.num_t_in * conv_comp.num_images
             by conv_comp.height_in * num_filters_in.  [highest-stride
             indexes come first in these multiplications].  It must
             satisfy input.NumCols() == input.Stride().
   @param [in] params   The parameters of the convolution.  This should be of
             dimension conv_comp.ParamRows() by conv_comp.ParamCols().
   @param [out] output   The output of the convolution (this function
             *adds to* the output).  Should be of dimension
             conv_comp.num_t_out * conv_comp.num_images
             by conv_comp.height_out * num_filters_out.  It must
             satisfy output.NumCols() == output.Stride().
 */
void ConvolveForward(
    const ConvolutionComputation &conv_comp,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &params,
    CuMatrixBase<BaseFloat> *output);


/**
   \brief This does the part of the backward derivative computation
          of convolution, that propagates derivatives back to
          the input data.  See also ConvolveBackwardParams(), which
          is for the parameter derivative.

   @param [in] conv_comp  A struct that describes the convolution
                          computation (should be the same as in
                          the corresponding forward pass).
   @param [in] params The parameters used in the forward convolution.  This
             should be of dimension num_filters_out by (X * num_filters_in),
             where X is the total number of pixels in the patches, which equals
             model.offsets.size() in the model for which the computation was
             compiled.  E.g. for a regular 3x3 kernel, X would be 9.
  @param [in] output_deriv The derivative of the objective function w.r.t. the
             output of the convolution.  Should be of dimension
             conv_comp.num_t_out * conv_comp.num_images by conv_comp.height_out
             * num_filters_out.  It must satisfy output_deriv.NumCols() ==
             output_deriv.Stride().
   @param [out] input_deriv  If non-NULL, the backpropagated derivative of
             the objective function w.r.t. the input will be *added to* this
             matrix.  Should be the same dimension as the input to the original
             ConvolveForward() call.
*/
void ConvolveBackwardData(
    const ConvolutionComputation &conv_comp,
    const CuMatrixBase<BaseFloat> &params,
    const CuMatrixBase<BaseFloat> &output_deriv,
    CuMatrixBase<BaseFloat> *input_deriv);

/**
   \brief This does the part of the backward derivative computation
          of convolution, that computes derivatives w.r.t. the
          parameters.  See also ConvolveBackwardData(), which computes
          derivatives w.r.t. the input data.

   @param [in] conv_comp  A struct that describes the computation
                          that was performed in the forward pass.
   @param [in] input     The input to the original forward convolution.  This
             should be of dimension (or should be reshapable to
             the dimension) conv_comp.num_t_in * conv_comp.num_images
             by conv_comp.height_in * num_filters_in.  [highest-stride
             indexes come first in these multiplications].  It must
             satisfy input.NumCols() == input.Stride().
   @param [in] output_deriv The derivative of the objective function w.r.t. the
             output of the convolution.  Should be of dimension
             conv_comp.num_t_out * conv_comp.num_images by conv_comp.height_out
             * num_filters_out.  It must satisfy output_deriv.NumCols() ==
             output_deriv.Stride().
   @param [in] alpha   This scalar is multiplied into the derivative when
             we add to params_deriv, i.e. *params_deriv += alpha * derivative.
   @param [out] params_deriv  The derivative of the objective function
             w.r.t the parameters (the 'params' given to the ConvolveForward
             function) is *added* to this location.  This matrix should be
             of dimension conv_comp.NumRows() by conv_comp.NumCols().
*/
void ConvolveBackwardParams(
    const ConvolutionComputation &conv_comp,
    const CuMatrixBase<BaseFloat> &input,
    const CuMatrixBase<BaseFloat> &output_deriv,
    BaseFloat alpha,
    CuMatrixBase<BaseFloat> *params_deriv);


/**
   This function takes lists of input and output indexes to a computation
   (e.g. as supplied to ReorderIndexes()), and figures out a regular structure
   for them (i.e. the smallest grid that will completely cover all the t,n
   pairs).
*/
void GetComputationIo(
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    ConvolutionComputationIo *io);


/**
   This function computes the reordered and possibly padded indexes
   corresponding to the computation in 'io'.  Note: the computation may have
   undergone various manipulations (padding, etc.) after being obtained by the
   function GetComputationIo().  The original input and output indexes are
   needed because they dictate the set of (n, x) pairs; and because they
   determine when to use 'real' indexes and when to use 'blank' padding values
   (i.e. when to replace the t values in the indexes by kNoTime).
*/
void GetIndexesForComputation(
    const ConvolutionComputationIo &io,
    const std::vector<Index> &orig_input_indexes,
    const std::vector<Index> &orig_output_indexes,
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes);


/**
   This function extends the set of input indexes that the computation
   has, to account for any required zero-padding in the time dimension.
   It reads model.all_time_offsets and model.time_offsets_modulus;
   and it may modify members start_t_in t_stride_in and num_t_in of *io.

   This is stage 1 of compilation.
 */
void PadComputationInputTime(const ConvolutionModel &model,
                             ConvolutionComputationIo *io);


/**
  This function takes a model that might require zero padding
  in the height dimension and outputs a model accepting a
  possibly-larger input dimension which does not require zero
  padding. *model_padded may differ from 'model' in its height_in and its
  'offsets' variable (the height-offsets need to be shifted if we pad at the
  bottom).  We then work out the computation in terms of the model that doesn't
  need padding (which is easier), and later convert it back to work in the space
  where there is no padding.

   This is stage 2 of compilation.
 */
void PadModelHeight(const ConvolutionModel &model,
                    ConvolutionModel *model_padded);


/** This function modifies, if necessary, a computation that has been built for
    the model 'model_padded', so that it can work for the original model
    'model'.  This may involve modifying the members 'height_in', 'temp_cols',
    and the column-related members of the elements of the 'steps' array.
    View it as the reverse step for 'PadModelHeight'.

    This function has to be aware that the computation will have been compiled
    after 'AppendInputFrames()' was called [this makes a difference in setups
    with subsampling], so the computation may have been built for input
    frames that were appended over several of the frames that 'model_padded'
    would require.

    This is the reverse step for stage 2 of compilation (it's a transformation
     of the computation).
 */
void UnPadModelHeight(const ConvolutionComputationOptions &opts,
                      const ConvolutionModel &model,
                      const ConvolutionModel &model_padded,
                      ConvolutionComputation *computation);

/**
   This function takes an input model and I/O specification, and it modifies
   both of them if necessary to ensure that the output 'io_appended' object has
   the same input and output time strides (i.e. t_stride_in == t_stride_out).
   This is done by appending the input frames across several time values and
   viewing them as single frames of larger dimension.

   The reason why 'io' is non-const is that it may be necessary to pad the
   number of input frames to ensure that the number of input frames is divisible
   by a multiple of t_stride_out / t_stride_in (if we pad the input frames, we
   pad to the right).

   The model in 'model_appended' may have larger height_in, and
   different values of 'offsets' and derived variables thereof, versus the model
   in 'model'.

   This is stage 3 of compilation.
*/
void AppendInputFrames(const ConvolutionModel &model,
                       ConvolutionComputationIo *io,
                       ConvolutionModel *model_appended,
                       ConvolutionComputationIo *io_appended);


/*
  This function takes a model and a specification of the comptuation's
  IO, and generates the computation.  This is stage 4 of the compilation.
  It assumes that stages 1, 2 and 3 have already been done so that:

    - Any required padding of the time axis (stage 1) and the height axis
      (stage 2) have been done (so any desired input values are available).
    - The t_stride_in and t_stride_out of the io object have the same value
      (stage 3).

  At this point the compilation process is actually quite simple: for each
  time shift (where the number of time shifts equals num_t_in + 1 - num_t_out
  of 'io'), we do a computation that copies and maybe duplicates the input
  columns to a temporary matrix, and then does a matrix multiplication
  between that temporary matrix
 */
void MakeComputation(const ConvolutionModel &model,
                     ConvolutionComputationIo &io,
                     const ConvolutionComputationOptions &opts,
                     ConvolutionComputation *computation);


} // namespace time_height_convolution

} // namespace nnet3





} // namespace kaldi


#endif
