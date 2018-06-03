/*
 * LHashTrie.cc --
 *	Instantiate trie class using LHash.
 *
 */

#ifndef _LHashTrie_cc_
#define _LHashTrie_cc_

#ifndef lint
static char LHashTrie_Copyright[] = "Copyright (c) 1997, SRI International.  All Rights Reserved.";
static char LHashTrie_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/LHashTrie.cc,v 1.1 1998/07/29 08:38:01 stolcke Exp $";
#endif

#ifdef USE_SARRAY_TRIE
#undef USE_SARRAY_TRIE
#include "Trie.cc"
#endif

#endif /* _LHashTrie_cc_ */
