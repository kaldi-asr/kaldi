// $Id: mempool.h 383 2010-04-23 15:29:28Z nicolabertoldi $

/******************************************************************************
 IrstLM: IRST Language Model Toolkit
 Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

******************************************************************************/

// An efficient memory manager
// by M. Federico
// Copyright Marcello Federico, ITC-irst, 1998

#ifndef MF_MEMPOOL_H
#define MF_MEMPOOL_H

#ifndef NULL
const int NULL=0;
#endif

#include <iostream>  // std::ostream

//! Memory block
/*! This can be used by:
- mempool to store items of fixed size
- strstack to store strings of variable size
*/


#define MP_BLOCK_SIZE 1000000
	
class memnode
{
  friend class mempool;   //!< grant access
  friend class strstack;  //!< grant access
  char          *block;   //!< block of memory
  memnode        *next;   //!< next block ptr
public:
  //! Creates a memory node
  memnode():block(NULL), next(NULL){};
	
  //! Destroys memory node
  ~memnode(){};
};


//! Memory pool

/*! A memory pool is composed of:
   - a linked list of block_num memory blocks
   - each block might contain up to block_size items
   - each item is made of exactly item_size bytes
*/

class mempool
{
  int         block_size;         //!< number of entries per block
  int         item_size;          //!< number of bytes per entry
  int         true_size;          //!< number of bytes per block
  memnode*   block_list;          //!< list of blocks
  char*       free_list;          //!< free entry list
  int         entries;            //!< number of stored entries
  int         blocknum;           //!< number of allocated blocks
public:

  //! Creates a memory pool
  mempool(int is, int bs=MP_BLOCK_SIZE);

  //! Destroys memory pool
  ~mempool();

  //! Prints a map of memory occupancy
  void map(std::ostream& co);

  //! Allocates a single memory entry
  char *allocate();

  //! Frees a single memory entry
  int free(char* addr);

  //! Prints statistics about this mempool
  void stat();

  //! Returns effectively used memory (bytes)
  /*! includes 8 bytes required by each call of new */

  int used() const {
    return blocknum * (true_size + 8);
  }

  //! Returns amount of wasted memory (bytes)
  int wasted() const {
    return used()-(entries * item_size);
  }
};

//! A stack to store strings

/*!
  The stack is composed of
  - a list of blocks memnode of fixed size
  - attribute blocknum tells the block on top
  - attribute idx tells position of the top string
*/

class strstack
{
  memnode* list; //!< list of memory blocks
  int   size;    //!< size of each block
  int    idx;    //!< index of last stored string
  int  waste;    //!< current waste of memory
  int memory;    //!< current use of memory
  int entries;   //!< current number of stored strings
  int blocknum;  //!< current number of used blocks

public:

  strstack(int bs=1000);

  ~strstack();

  const char *push(const char *s);

  const char *pop();

  const char *top();

  void stat();

  int used() const {
    return memory;
  }

  int wasted() const {
    return waste;
  }

};


//! Manages multiple memory pools

/*!
  This class permits to manage memory pools
  with items up to a specified size.
  - items within the allowed range are stored in memory pools
  - items larger than the limit are allocated with new
*/


class storage
{
  mempool **poolset;  //!< array of memory pools
  int setsize;        //!< number of memory pools/maximum elem size
  int poolsize;       //!< size of each block
  int newmemory;      //!< stores amount of used memory
  int newcalls;       //!< stores number of allocated blocks
public:

  //! Creates storage
  storage(int maxsize,int blocksize);

  //! Destroys storage
  ~storage();

  /* names of below functions have been changed so as not to interfere with macros for malloc/realloc/etc -- EVH */

  //! Allocates memory
  char *allocate(int size);

  //! Realloc memory
  char *reallocate(char *oldptr,int oldsize,int newsize);

  //! Frees memory of an entry
  int free(char *addr,int size=0);

  //! Prints statistics about storage
  void stat();
};

#endif
