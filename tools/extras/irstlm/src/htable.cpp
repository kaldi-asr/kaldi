// $Id: htable.cpp 3680 2010-10-13 09:10:21Z bertoldi $

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "mempool.h"
#include "htable.h"
#include "util.h"

using namespace std;

template <>
void htable<int*>::set_keylen(int kl)
{
  keylen=kl/sizeof(int);
  return;
}

template <>
void htable<char*>::set_keylen(int kl)
{
  keylen=kl;
  return;
}

template <>
address htable<int *>::Hash(int* key)
{
  address  h;
  int i;

  //Thomas Wang's 32 bit Mix Function
  for (i=0,h=0; i<keylen; i++) {
    h+=key[i];
    h += ~(h << 15);
    h ^=  (h >> 10);
    h +=  (h << 3);
    h ^=  (h >> 6);
    h += ~(h << 11);
    h ^=  (h >> 16);
  };

  return h;
}

template <>
address htable<char *>::Hash(char* key)
{
  //actually char* key is a char**, i.e. a pointer to a char*
  char *Key = *(char**)key;
  int  length=strlen(Key);

  address h=0;
  int i;

  for (i=0,h=0; i<length; i++)
    h = h * Prime1 ^ (Key[i] - ' ');
  h %= Prime2;

  return h;
}

template <>
int htable<int*>::Comp(int *key1, int *key2) const
{
  MY_ASSERT(key1 && key2);

  int i;

  for (i=0; i<keylen; i++)
    if (key1[i]!=key2[i]) return 1;
  return 0;
}

template <>
int htable<char*>::Comp(char *key1, char *key2) const
{
  MY_ASSERT(key1 && key2);

  char *Key1 = *(char**)key1;
  char *Key2 = *(char**)key2;

  MY_ASSERT(Key1 && Key2);

  return (strcmp(Key1,Key2));
}
