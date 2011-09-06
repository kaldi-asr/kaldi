// util/table-types.h

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


#ifndef KALDI_UTIL_TABLE_TYPES_H_
#define KALDI_UTIL_TABLE_TYPES_H_
#include "base/kaldi-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

// This header defines typedefs that are specific instantiations of
// the Table types.

/// \addtogroup table_types
/// @{

typedef TableWriter<KaldiObjectHolder<Matrix<BaseFloat> > >  BaseFloatMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<Matrix<BaseFloat> > >  SequentialBaseFloatMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Matrix<BaseFloat> > >  RandomAccessBaseFloatMatrixReader;

typedef TableWriter<KaldiObjectHolder<Matrix<double> > >  DoubleMatrixWriter;
typedef SequentialTableReader<KaldiObjectHolder<Matrix<double> > >  SequentialDoubleMatrixReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Matrix<double> > >  RandomAccessDoubleMatrixReader;

typedef TableWriter<KaldiObjectHolder<Vector<BaseFloat> > >  BaseFloatVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<Vector<BaseFloat> > >  SequentialBaseFloatVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Vector<BaseFloat> > >  RandomAccessBaseFloatVectorReader;

typedef TableWriter<KaldiObjectHolder<Vector<double> > >  DoubleVectorWriter;
typedef SequentialTableReader<KaldiObjectHolder<Vector<double> > >  SequentialDoubleVectorReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Vector<double> > >  RandomAccessDoubleVectorReader;


typedef TableWriter<BasicHolder<int32> >  Int32Writer;
typedef SequentialTableReader<BasicHolder<int32> >  SequentialInt32Reader;
typedef RandomAccessTableReader<BasicHolder<int32> >  RandomAccessInt32Reader;

typedef TableWriter<BasicVectorHolder<int32> >  Int32VectorWriter;
typedef SequentialTableReader<BasicVectorHolder<int32> >  SequentialInt32VectorReader;
typedef RandomAccessTableReader<BasicVectorHolder<int32> >  RandomAccessInt32VectorReader;

typedef TableWriter<BasicVectorVectorHolder<int32> >  Int32VectorVectorWriter;
typedef SequentialTableReader<BasicVectorVectorHolder<int32> >  SequentialInt32VectorVectorReader;
typedef RandomAccessTableReader<BasicVectorVectorHolder<int32> >  RandomAccessInt32VectorVectorReader;

typedef TableWriter<BasicPairVectorHolder<int32> >  Int32PairVectorWriter;
typedef SequentialTableReader<BasicPairVectorHolder<int32> >  SequentialInt32PairVectorReader;
typedef RandomAccessTableReader<BasicPairVectorHolder<int32> >  RandomAccessInt32PairVectorReader;

typedef TableWriter<BasicHolder<BaseFloat> >  BaseFloatWriter;
typedef SequentialTableReader<BasicHolder<BaseFloat> >  SequentialBaseFloatReader;
typedef RandomAccessTableReader<BasicHolder<BaseFloat> >  RandomAccessBaseFloatReader;

typedef TableWriter<BasicHolder<double> >  DoubleWriter;
typedef SequentialTableReader<BasicHolder<double> >  SequentialDoubleReader;
typedef RandomAccessTableReader<BasicHolder<double> >  RandomAccessDoubleReader;

typedef TableWriter<BasicHolder<bool> >  BoolWriter;
typedef SequentialTableReader<BasicHolder<bool> >  SequentialBoolReader;
typedef RandomAccessTableReader<BasicHolder<bool> >  RandomAccessBoolReader;



/// TokenWriter is a writer specialized for std::string where the strings
/// are nonempty and whitespace-free.   T == std::string
typedef TableWriter<TokenHolder> TokenWriter;
typedef SequentialTableReader<TokenHolder> SequentialTokenReader;
typedef RandomAccessTableReader<TokenHolder> RandomAccessTokenReader;


/// TokenVectorWriter is a writer specialized for sequences of
/// std::string where the strings are nonempty and whitespace-free.
/// T == std::vector<std::string>
typedef TableWriter<TokenVectorHolder> TokenVectorWriter;
// Ditto for SequentialTokenVectorReader.
typedef SequentialTableReader<TokenVectorHolder> SequentialTokenVectorReader;
typedef RandomAccessTableReader<TokenVectorHolder> RandomAccessTokenVectorReader;


// Posterior is a typedef: vector<vector<pair<int32, BaseFloat> > >,
// representing posteriors over (typically) transition-ids for an
// utterance.
typedef TableWriter<PosteriorHolder> PosteriorWriter;
typedef SequentialTableReader<PosteriorHolder> SequentialPosteriorReader;
typedef RandomAccessTableReader<PosteriorHolder> RandomAccessPosteriorReader;


// typedef std::vector<std::vector<std::pair<int32, Vector<BaseFloat> > > > GauPost;
typedef TableWriter<GauPostHolder> GauPostWriter;
typedef SequentialTableReader<GauPostHolder> SequentialGauPostReader;
typedef RandomAccessTableReader<GauPostHolder> RandomAccessGauPostReader;

/// @}

// Note: for FST reader/writer, see ../fstext/fstext-utils.h
// [not done yet].

} // end namespace kaldi


#endif
