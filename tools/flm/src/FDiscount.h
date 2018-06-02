/*
 * FDiscount.h --
 *	Discounting schemes for factored LMs
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 *
 *
 * Copyright (c) 1995-2010 SRI International.  All Rights Reserved.
 *
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/FDiscount.h,v 1.10 2012/10/20 00:22:25 mcintyre Exp $
 *
 *
 */

#ifndef _FDiscount_h_
#define _FDiscount_h_

#include "Discount.h"

#include "FNgramStats.h"
#include "FNgramSpecs.h"
#include "FactoredVocab.h"

/*
 * FDiscount --
 *	Methods to manipulate factored counts for estimation purposes.
 */
class FDiscount: public virtual Discount
{
public:
    // Note: replicate default dummy versions here to avoid warnings from 
    // some compilers

    virtual Boolean estimate(NgramStats &counts, unsigned order)
        { return false; };	/* can't do it */

    // support for factored models

    /*
     * dummy estimator for when there is nothing to estimate
     */
    virtual Boolean estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
	{ return true; };

    /*
     * dummy estimator for when there is nothing to prepare
     */
    virtual void prepareCounts(NgramCounts<NgramCount> &counts,
				unsigned order, unsigned maxOrder)
	{ return; };
    virtual void prepareCounts(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab)
	{ return; };
};


/*
 * FGoodTuring --
 *	The standard discounting method based on count of counts
 */
class FGoodTuring: public FDiscount, public GoodTuring 
{
public:
    FGoodTuring(unsigned mincount = GT_defaultMinCount,
                unsigned maxcount = GT_defaultMaxCount)
	: GoodTuring(mincount, maxcount) {};


    double discount(Count count, Count totalCount, Count observedVocab) {
	return GoodTuring::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
			    Count min2Vocab, Count min3Vocab) {
	return GoodTuring::lowerOrderWeight(totalCount,observedVocab,
					    min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return GoodTuring::nodiscount(); }
    void write(File &file) { return GoodTuring::write(file); }
    Boolean read(File &file) { return GoodTuring::read(file); }

    Boolean estimate(NgramStats &counts, unsigned order)
        { return false; };	/* can't do it */
    Boolean estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
			unsigned int node, FactoredVocab& vocab);
};


/*
 * FConstDiscount --
 *      Ney's method of subtracting a constant <= 1 from all counts
 *      (also known as "Absolute discounting").
 */
class FConstDiscount: public FDiscount, public ConstDiscount 
{
public:
    FConstDiscount(double d, unsigned mincount = 0)
	: ConstDiscount(d, mincount) {};

    double discount(Count count, Count totalCount, Count observedVocab) {
	return ConstDiscount::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
			    Count min2Vocab, Count min3Vocab) {
	return ConstDiscount::lowerOrderWeight(totalCount,observedVocab,
					    min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return ConstDiscount::nodiscount(); }
    void write(File &file) { return ConstDiscount::write(file); }
    Boolean read(File &file) { return ConstDiscount::read(file); }
};


/*
 * FNaturalDiscount --
 *	Ristad's natural law of succession
 */
class FNaturalDiscount: public FDiscount, public NaturalDiscount 
{
public:
    FNaturalDiscount(unsigned mincount = 0) : NaturalDiscount(mincount) {};

    double discount(Count count, Count totalCount, Count observedVocab) {
	return NaturalDiscount::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
			    Count min2Vocab, Count min3Vocab) {
	return NaturalDiscount::lowerOrderWeight(totalCount,observedVocab,
						 min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return NaturalDiscount::nodiscount(); }
    void write(File &file) { return NaturalDiscount::write(file); }
    Boolean read(File &file) { return NaturalDiscount::read(file); }

    Boolean estimate(NgramStats &counts, unsigned order)
        { return false; };	/* can't do it */
    Boolean estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab);
};


/*
 * FWittenBell --
 *      Witten & Bell's method of estimating the probability of an
 *      unseen event by the total number of 'new' events overserved,
 *      i.e., counting each observed word type once.
 */
class FWittenBell: public FDiscount, public WittenBell 
{
public:
    FWittenBell(unsigned mincount = 0) : WittenBell(mincount) {};

    double discount(Count count, Count totalCount, Count observedVocab) {
	return WittenBell::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
			    Count min2Vocab, Count min3Vocab) {
	return WittenBell::lowerOrderWeight(totalCount,observedVocab,
					    min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return WittenBell::nodiscount(); }
    void write(File &file) { return WittenBell::write(file); }
    Boolean read(File &file) { return WittenBell::read(file); }
};


/*
 * FKneserNey --
 *	Regular Kneser-Ney discounting
 */
class FKneserNey: public FDiscount, public KneserNey 
{
public:
    FKneserNey(unsigned mincount = 0,
		Boolean countsAreModified = false,
		Boolean prepareCountsAtEnd = false)
	: KneserNey(mincount, countsAreModified, prepareCountsAtEnd) {};

    double discount(Count count, Count totalCount, Count observedVocab) {
	return KneserNey::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
 		  	    Count min2Vocab, Count min3Vocab) {
	return KneserNey::lowerOrderWeight(totalCount,observedVocab,
				  	   min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return KneserNey::nodiscount(); }
    void write(File &file) { return KneserNey::write(file); }
    Boolean read(File &file) { return KneserNey::read(file); }

    virtual Boolean estimate(NgramStats &counts, unsigned order)
        { return false; };	/* can't do it */
    virtual Boolean estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab);
    virtual void prepareCounts(NgramCounts<NgramCount> &counts,
				unsigned order, unsigned maxOrder)
	{ return; };
    virtual void prepareCounts(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab);
};


/*
 * FModKneserNey --
 *	Modified Kneser-Ney discounting (Chen & Goodman 1998)
 */
class FModKneserNey: public FKneserNey, public ModKneserNey 
{
public:
    FModKneserNey(unsigned mincount = 0,
		  Boolean countsAreModified = false,
		  Boolean prepareCountsAtEnd = false)
    	: FKneserNey(mincount, countsAreModified, prepareCountsAtEnd),
	  ModKneserNey(mincount, countsAreModified, prepareCountsAtEnd) {};

    double discount(Count count, Count totalCount, Count observedVocab) {
	return ModKneserNey::discount(count,totalCount,observedVocab);
    }
    double lowerOrderWeight(Count totalCount, Count observedVocab,
			    Count min2Vocab, Count min3Vocab) {
	return ModKneserNey::lowerOrderWeight(totalCount,observedVocab,
					      min2Vocab,min3Vocab);
    }
    Boolean nodiscount() { return ModKneserNey::nodiscount(); }
    void write(File &file) { return ModKneserNey::write(file); }
    Boolean read(File &file) { return ModKneserNey::read(file); }

    virtual Boolean estimate(NgramStats &counts, unsigned order)
        { return false; };	/* can't do it */
    Boolean estimate(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
				unsigned int node, FactoredVocab& vocab);

    virtual void prepareCounts(FNgramSpecs<FNgramCount>::FNgramSpec &spec,
			       unsigned int node, FactoredVocab& vocab)
	{ return FKneserNey::prepareCounts(spec,node,vocab); }
};

#endif /* _FDiscount_h_ */

