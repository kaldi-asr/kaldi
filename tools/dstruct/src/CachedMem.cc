/*
 * CachedMem.cc --
 *      A simple memory management template class 
 *
 */

#ifndef _CachedMem_cc_
#define _CachedMem_cc_

#ifndef lint
static char CachedMem_Copyright[] = "Copyright (c) 2008-2012 SRI International.  All Rights Reserved.";
static char CachedMem_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/CachedMem.cc,v 1.7 2015-04-17 00:33:01 frandsen Exp $";
#endif

#include "CachedMem.h"

#include <iostream>
using namespace std;

template <class T>
TLSW_DEF(T *, CachedMem<T>::__freelistTLS);
template <class T>
TLSW_DEF(CachedMemUnit *, CachedMem<T>::__alloclistTLS);
template <class T>
TLSW_DEF(int, CachedMem<T>::__num_newTLS);
template <class T>
TLSW_DEF(int, CachedMem<T>::__num_delTLS);
template <class T>
TLSW_DEF(int, CachedMem<T>::__num_chkTLS);

template <class T> 
const size_t CachedMem<T>::__chunk = 64;

template <class T>
void CachedMem<T>:: freeall ()
{
  CachedMemUnit* &__alloclist = TLSW_GET(__alloclistTLS);
  T* &__freelist = TLSW_GET(__freelistTLS);
  int &__num_chk = TLSW_GET(__num_chkTLS);
  int &__num_new = TLSW_GET(__num_newTLS);
  int &__num_del = TLSW_GET(__num_delTLS);

  while (__alloclist) {
    CachedMemUnit * p = __alloclist->next;
    ::delete [] static_cast<T*>(__alloclist->mem);
    __alloclist = p;

  }
  __freelist = 0;
  
  __num_chk = 0;
  __num_new = 0;
  __num_del = 0;
}

template <class T>
void CachedMem<T>::freeThread()
{
    freeall();

    TLSW_FREE(__alloclistTLS);
    TLSW_FREE(__freelistTLS);
    TLSW_FREE(__num_chkTLS);
    TLSW_FREE(__num_newTLS);
    TLSW_FREE(__num_delTLS);
}

template <class T> 
void CachedMem<T>:: stat() 
{
  int &__num_chk = TLSW_GET(__num_chkTLS);
  int &__num_new = TLSW_GET(__num_newTLS);
  int &__num_del = TLSW_GET(__num_delTLS);

  cerr << "Number of allocated chunks: " << __num_chk << endl;
  cerr << "Number of \"new\" calls: " << __num_new  << endl;
  cerr << "Number of \"delete\" calls: " << __num_del << endl;
}

template <class T> 
void * CachedMem<T>::operator new (size_t sz)
{
  CachedMemUnit* &__alloclist = TLSW_GET(__alloclistTLS);
  T* &__freelist = TLSW_GET(__freelistTLS);
  int &__num_chk = TLSW_GET(__num_chkTLS);
  int &__num_new = TLSW_GET(__num_newTLS);

  if (sz != sizeof(T)) 
#ifdef NO_EXCEPTIONS
    return 0;
#else
    throw "CachedMem: size mismatch in operator new";
#endif

  __num_new ++;
  
  if (!__freelist) {
    
    T * array = ::new T [ __chunk ];
    for (size_t i = 0; i < __chunk; i++)
      addToFreelist(&array[i]);    
    
    CachedMemUnit * u = ::new CachedMemUnit;
    u->next = __alloclist;
    u->mem = static_cast<void *> (array);
    __alloclist = u;

    __num_chk ++;
  }

  T * p = __freelist;
  if (__freelist) {
    __freelist = __freelist->CachedMem<T>::__next;
  } // else unexpected error (addToFreelist failed)
  return p;
}

#endif /* _CachedMem_cc_ */

