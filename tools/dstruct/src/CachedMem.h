/*
 * CachedMem --
 *      A simple memory management template class 
 *
 *  Copyright (c) 2012 SRI International. All Rights Reserved.
 * 
 */

#ifndef _CachedMem_h_
#define _CachedMem_h_

#include <stdio.h>

#include "TLSWrapper.h"

struct CachedMemUnit { 
  CachedMemUnit * next;
  void * mem;
};

template <class T> 
class CachedMem {

public:
  virtual ~CachedMem () {};
  void * operator new (size_t);
  void   operator delete (void * p, size_t) {
    int &__num_del = TLSW_GET(__num_delTLS);
    if (p) addToFreelist(static_cast<T*>(p));
    __num_del ++;
  }
  static void  freeall();
  static void  stat();
  static void  freeThread();

protected:
  T * __next;

private:
  static void addToFreelist(T* p) {
    T* &__freelist = TLSW_GET(__freelistTLS);
    ((CachedMem<T> *) p)->__next = __freelist;
    __freelist = p;
  }

  static TLSW_DECL(T *, __freelistTLS);
  static TLSW_DECL(CachedMemUnit *, __alloclistTLS); 
  static const size_t __chunk;

  // statistics 
  static TLSW_DECL(int, __num_newTLS);
  static TLSW_DECL(int, __num_delTLS);
  static TLSW_DECL(int, __num_chkTLS);
  
};

#endif /* _CachedMem_h_ */

