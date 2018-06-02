/*
 * Map.cc --
 *	Map implementation
 *
 */

#ifndef lint
static char Map_Copyright[] = "Copyright (c) 1995, SRI International.  All Rights Reserved.";
static char Map_RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/dstruct/src/Map.cc,v 1.5 2012/10/11 20:23:52 mcintyre Exp $";
#endif

#include "Map.h"

unsigned int _Map::initialSize = 10;
float _Map::growSize = 1.5;

